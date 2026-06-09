using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("EXPERIMENTAL decoupled God node. Acquisition thread casts U16->F32, decimates and z-scores " +
                 "into a lock-free ring (+ parallel clock ring); a pinned RealTime background thread skips to the " +
                 "freshest window, runs zero-alloc ONNX + FSM, and emits RippleOut. Inference is skipped during the " +
                 "TTL refractory. Accepts a (data, clock) Zip so detections carry the hardware sample clock.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorSuperNode : IDisposable
    {
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        const uint _MCW_DN = 0x03000000;
        const uint _DN_FLUSH = 0x01000000;

        [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod")] private static extern uint TimeBeginPeriod(uint ms);
        [DllImport("winmm.dll", EntryPoint = "timeEndPeriod")] private static extern uint TimeEndPeriod(uint ms);
        [DllImport("kernel32.dll")] static extern IntPtr GetCurrentThread();
        [DllImport("kernel32.dll")] static extern IntPtr SetThreadAffinityMask(IntPtr hThread, IntPtr dwThreadAffinityMask);

        // --- 1. Model ---
        [Category("1. Model")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";
        [Category("1. Model")] public int TimePoints { get; set; } = 44;
        [Category("1. Model")] public int Channels { get; set; } = 8;

        // --- 2. Signal Timing ---
        [Category("2. Signal Timing")]
        [Description("Downsample factor on the incoming full-rate stream (e.g. 12 => 30kHz -> 2.5kHz).")]
        public int DecimationFactor { get; set; } = 12;
        [Category("2. Signal Timing")]
        [Description("Run inference whenever this many new decimated samples have arrived (stride). 3 => Buffer(44,3).")]
        public int InferenceStride { get; set; } = 3;

        // --- 3. Adaptive Z-Score ---
        [Category("3. Adaptive Z-Score")] public int BaselineWindowSize { get; set; } = 1250;
        [Category("3. Adaptive Z-Score")]
        [Description("Scale applied to the z-scored value before the model. Match training (1.0 = pure z).")]
        public float ZScoreOutputScale { get; set; } = 1.0f;

        // --- 4. Thresholds ---
        [Category("4. Thresholds")] public float Threshold1_IgnoreBelow { get; set; } = 0.10f;
        [Category("4. Thresholds")] public float Threshold2_WeakEvidence { get; set; } = 0.50f;
        [Category("4. Thresholds")] public float Threshold3_StrongEvidence { get; set; } = 0.80f;
        [Category("4. Thresholds")] public float TargetEvidenceScore { get; set; } = 5.0f;

        // --- 5. Anti-Flicker ---
        [Category("5. Anti-Flicker")] public int FramesToWaitBeforeDecay { get; set; } = 5;
        [Category("5. Anti-Flicker")] public float ScoreDecayPerFrame { get; set; } = 1.0f;

        // --- 6. Hardware TTL ---
        [Category("6. Hardware TTL")]
        [Description("Refractory hold after a detection. Inference is SKIPPED during this window (free headroom).")]
        public int PostRippleRefractoryMs { get; set; } = 150;

        // --- 7. Hardware Tuning ---
        [Category("7. Hardware Tuning")]
        [Description("Pin the inference thread to this core. -1 disables. Busy-spin pegs it.")]
        public int TargetCore { get; set; } = 7;
        [Category("7. Hardware Tuning")]
        [Description("Busy-spin for lowest pickup latency (pegs TargetCore). False = SpinOnce.")]
        public bool BusySpin { get; set; } = true;
        [Category("7. Hardware Tuning")]
        public ProcessPriorityClass ProcessPriority { get; set; } = ProcessPriorityClass.RealTime;

        // Ring of z-scored decimated samples + parallel ring of sample clocks.
        private const int RingSize = 1024;
        private float[] _ring;
        private GCHandle _hRing;
        private ulong[] _clockRing;
        private volatile int _writeHead;
        private long _written;
        private long _sampleIndex;
        private int _channels, _timePoints, _baselineWindow;

        // Z-score state (writer thread only)
        private float[] _zHistory;
        private double[] _zSum, _zSumSq;
        private int _zWriteIndex, _zSamplesSeen, _decimationCounter;

        // FSM state (background thread only)
        private RippleState _fsmState = RippleState.NoRipple;
        private float _scoreTicks;
        private int _ticksDipping, _eventCount;
        private bool _ttlHolding;
        private long _ttlHoldUntilMs;
        private static readonly Stopwatch Clock = Stopwatch.StartNew();
        private RippleOut _currentOutput;

        private CancellationTokenSource _cts;

        // Single-stream input (clock falls back to a decimated-sample index).
        public IObservable<RippleOut> Process(IObservable<Mat> source) =>
            Run(observer => source.Subscribe(m => WriteBlock(m, null), observer.OnError));

        // Zipped (data, clock) input: detections carry the hardware sample clock (onix Clock is ulong[]).
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong[]>> source) =>
            Run(observer => source.Subscribe(t => WriteBlock(t.Item1, t.Item2), observer.OnError));

        private IObservable<RippleOut> Run(Func<IObserver<RippleOut>, IDisposable> subscribe)
        {
            return Observable.Create<RippleOut>(observer =>
            {
                _channels = Channels;
                _timePoints = TimePoints;
                _baselineWindow = BaselineWindowSize;

                _ring = new float[RingSize * _channels];
                _hRing = GCHandle.Alloc(_ring, GCHandleType.Pinned);
                _clockRing = new ulong[RingSize];
                _writeHead = 0; _written = 0; _sampleIndex = 0;

                _zHistory = new float[_channels * _baselineWindow];
                _zSum = new double[_channels];
                _zSumSq = new double[_channels];
                _zWriteIndex = _zSamplesSeen = _decimationCounter = 0;

                _fsmState = RippleState.NoRipple;
                _scoreTicks = 0; _ticksDipping = 0; _eventCount = 0; _ttlHolding = false;
                _currentOutput = new RippleOut { State = RippleState.NoRipple, TTL = false };
                Clock.Restart();

                _cts = new CancellationTokenSource();
                var sub = subscribe(observer);

                var worker = new Thread(() => RunInference(observer, _cts.Token))
                {
                    IsBackground = true,
                    Priority = ThreadPriority.Highest,
                    Name = "RippleSuperNodeInference"
                };
                worker.Start();

                return Disposable.Create(() =>
                {
                    _cts.Cancel();
                    worker.Join(500);
                    sub.Dispose();
                    if (_hRing.IsAllocated) _hRing.Free();
                });
            });
        }

        // ----- Writer thread: U16/F32 -> decimate -> rolling z-score -> ring (+ clock) -----
        private void WriteBlock(Mat data, ulong[] clock)
        {
            if (data.Rows < _channels)
                throw new InvalidOperationException(
                    $"Input has {data.Rows} rows but Channels={_channels}. Feed [Channels x N] channel-major.");

            int channels = _channels;
            int baseWin = _baselineWindow;
            int decim = DecimationFactor < 1 ? 1 : DecimationFactor;
            float zScale = ZScoreOutputScale;
            bool isU16 = data.Depth == Depth.U16;
            int elemSize = isU16 ? sizeof(ushort) : sizeof(float);
            int nCols = data.Cols;
            int step = data.Step;

            bool hasClock = clock != null;
            int clockLen = hasClock ? clock.Length : 0;

            unsafe
            {
                byte* dbase = (byte*)data.Data.ToPointer();
                float* ring = (float*)_hRing.AddrOfPinnedObject().ToPointer();

                for (int t = 0; t < nCols; t++)
                {
                    if (++_decimationCounter < decim) continue;
                    _decimationCounter = 0;

                    int n_t = _zSamplesSeen < 1 ? 1 : _zSamplesSeen;
                    int ringOff = _writeHead * channels;

                    for (int c = 0; c < channels; c++)
                    {
                        byte* p = dbase + c * step + t * elemSize;
                        float newValue = isU16 ? *(ushort*)p : *(float*)p;
                        int histOff = c * baseWin + _zWriteIndex;

                        float oldValue = (_zSamplesSeen < baseWin) ? 0f : _zHistory[histOff];
                        _zSum[c]   += newValue - oldValue;
                        _zSumSq[c] += (double)newValue * newValue - (double)oldValue * oldValue;
                        _zHistory[histOff] = newValue;

                        double mu = _zSum[c] / n_t;
                        double variance = Math.Max(0.0, _zSumSq[c] / n_t - mu * mu);
                        double sigma = Math.Max(1e-10, Math.Sqrt(variance));
                        ring[ringOff + c] = (float)(((newValue - mu) / sigma) * zScale);
                    }

                    _clockRing[_writeHead] = (hasClock && t < clockLen) ? clock[t] : (ulong)_sampleIndex;
                    _sampleIndex++;

                    _zWriteIndex = (_zWriteIndex + 1) % baseWin;
                    if (_zSamplesSeen < baseWin) _zSamplesSeen++;

                    _writeHead = (_writeHead + 1) & (RingSize - 1);  // volatile write publishes ring + clock
                    _written++;
                }
            }
        }

        // ----- Background thread: freshest-window inference + FSM (skipped during refractory) -----
        private unsafe void RunInference(IObserver<RippleOut> observer, CancellationToken token)
        {
            try { uint cc = 0; _controlfp_s(ref cc, _DN_FLUSH, _MCW_DN); } catch { }
            TimeBeginPeriod(1);
            try
            {
                using (var proc = System.Diagnostics.Process.GetCurrentProcess())
                    proc.PriorityClass = ProcessPriority;
            }
            catch { }
            if (TargetCore >= 0 && TargetCore < 64)
                SetThreadAffinityMask(GetCurrentThread(), (IntPtr)(1L << TargetCore));

            int windowFloats = _timePoints * _channels;
            int strideSamples = InferenceStride < 1 ? 1 : InferenceStride;

            var opts = new SessionOptions
            {
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                EnableCpuMemArena = true,
                EnableMemoryPattern = true
            };
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            opts.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");

            var session = new InferenceSession(ModelPath, opts);
            var binding = session.CreateIoBinding();
            var runOpts = new RunOptions();
            opts.Dispose();

            float[] inBuf = new float[windowFloats];
            var hIn = GCHandle.Alloc(inBuf, GCHandleType.Pinned);
            float[] outBuf = new float[1];
            var hOut = GCHandle.Alloc(outBuf, GCHandleType.Pinned);
            var mem = OrtMemoryInfo.DefaultInstance;
            var valIn = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(inBuf), new long[] { 1, _timePoints, _channels });
            var valOut = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(outBuf), new long[] { 1, 1 });
            binding.BindInput(session.InputMetadata.Keys.First(), valIn);
            binding.BindOutput(session.OutputMetadata.Keys.First(), valOut);
            for (int i = 0; i < 50; i++) session.RunWithBinding(runOpts, binding);

            float* ring = (float*)_hRing.AddrOfPinnedObject().ToPointer();
            float* dst = (float*)hIn.AddrOfPinnedObject().ToPointer();
            var spinner = new SpinWait();
            int lastInferHead = 0;
            int skipped = 0;

            try
            {
                while (!token.IsCancellationRequested)
                {
                    if (Volatile.Read(ref _written) < _timePoints)
                    {
                        if (BusySpin) Thread.SpinWait(64); else spinner.SpinOnce();
                        continue;
                    }

                    int head = _writeHead;
                    int newSamples = (head - lastInferHead) & (RingSize - 1);
                    if (newSamples < strideSamples)
                    {
                        if (BusySpin) Thread.SpinWait(64); else spinner.SpinOnce();
                        continue;
                    }

                    if (_ttlHolding)
                    {
                        // Refractory: advance the hold timer and keep emitting the held TTL. No ONNX.
                        RunFSM(0f);
                    }
                    else
                    {
                        if (newSamples > strideSamples * 2) skipped += (newSamples / strideSamples) - 1;

                        // Copy the freshest window [head - TimePoints, head), wrap-safe.
                        int start = (head - _timePoints) & (RingSize - 1);
                        int firstSamples = Math.Min(_timePoints, RingSize - start);
                        Buffer.MemoryCopy(ring + start * _channels, dst,
                            windowFloats * sizeof(float), firstSamples * _channels * sizeof(float));
                        if (firstSamples < _timePoints)
                        {
                            int remFloats = (_timePoints - firstSamples) * _channels;
                            Buffer.MemoryCopy(ring, dst + firstSamples * _channels,
                                remFloats * sizeof(float), remFloats * sizeof(float));
                        }

                        session.RunWithBinding(runOpts, binding);
                        RunFSM(outBuf[0]);
                    }

                    _currentOutput.StrideUsed = skipped;
                    _currentOutput.Clock = _clockRing[(head - 1) & (RingSize - 1)];
                    observer.OnNext(_currentOutput);
                    lastInferHead = head;
                }
            }
            finally
            {
                valIn.Dispose(); valOut.Dispose(); binding.Dispose(); runOpts.Dispose(); session.Dispose();
                if (hIn.IsAllocated) hIn.Free();
                if (hOut.IsAllocated) hOut.Free();
                TimeEndPeriod(1);
            }
        }

        private void RunFSM(float probability)
        {
            long now = Clock.ElapsedMilliseconds;

            if (_ttlHolding)
            {
                if (now >= _ttlHoldUntilMs)
                {
                    _ttlHolding = false;
                    _fsmState = RippleState.NoRipple;
                    _scoreTicks = 0; _ticksDipping = 0;
                }
                else { UpdateOutput(probability, true); return; }
            }

            switch (_fsmState)
            {
                case RippleState.NoRipple:
                    if (probability >= Threshold1_IgnoreBelow)
                    {
                        _fsmState = RippleState.Possible;
                        _scoreTicks = 0; _ticksDipping = 0;
                    }
                    break;

                case RippleState.Possible:
                    if (probability >= Threshold1_IgnoreBelow)
                    {
                        _ticksDipping = 0;
                        if (probability >= Threshold3_StrongEvidence) _scoreTicks += 2.0f;
                        else if (probability >= Threshold2_WeakEvidence) _scoreTicks += 1.0f;

                        if (_scoreTicks >= TargetEvidenceScore)
                        {
                            _fsmState = RippleState.Ripple;
                            _eventCount++;
                            _scoreTicks = 0; _ticksDipping = 0;
                            _ttlHolding = true;
                            _ttlHoldUntilMs = now + Math.Max(1, PostRippleRefractoryMs);
                            UpdateOutput(probability, true);
                            return;
                        }
                    }
                    else
                    {
                        _ticksDipping++;
                        if (_ticksDipping > FramesToWaitBeforeDecay)
                        {
                            _scoreTicks -= ScoreDecayPerFrame;
                            if (_scoreTicks <= 0)
                            {
                                _fsmState = RippleState.NoRipple;
                                _scoreTicks = 0; _ticksDipping = 0;
                            }
                        }
                    }
                    break;

                case RippleState.Ripple:
                    if (probability < Threshold1_IgnoreBelow)
                    {
                        _fsmState = RippleState.NoRipple;
                        _scoreTicks = 0; _ticksDipping = 0;
                    }
                    break;
            }

            UpdateOutput(probability, false);
        }

        private void UpdateOutput(float prob, bool ttl)
        {
            _currentOutput.Probability = prob;
            _currentOutput.TTL = ttl;
            _currentOutput.State = _fsmState;
            _currentOutput.Score = _scoreTicks;
            _currentOutput.EventCount = _eventCount;
        }

        public void Dispose()
        {
            _cts?.Cancel();
        }
    }
}
