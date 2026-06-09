using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("God Node: block-rate raw input -> internal decimation -> rectangular adaptive Z-Score " +
                 "-> linear sliding-window ONNX -> Leaky Bucket FSM. Zero per-sample allocation.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorAdaptive : IDisposable
    {
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        const uint _MCW_DN = 0x03000000;
        const uint _DN_FLUSH = 0x01000000;

        // --- UI CATEGORY 1: Neural Network ---
        [Category("1. Neural Network")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("1. Neural Network")] public bool UseCUDA { get; set; } = false;
        [Category("1. Neural Network")] public int DeviceId { get; set; } = 0;

        // --- UI CATEGORY 2: Dimensions & Timing ---
        [Category("2. Signal Timing")] public int Channels { get; set; } = 8;
        [Category("2. Signal Timing")] public int TimePoints { get; set; } = 44;

        [Category("2. Signal Timing")]
        [Description("Downsample factor applied to the incoming full-rate stream. (e.g., 12 turns 30kHz into 2.5kHz)")]
        public int DecimationFactor { get; set; } = 12;

        [Category("2. Signal Timing")]
        [Description("How many decimated samples to wait before running inference again. (e.g., 2 or 3)")]
        public int InferenceStride { get; set; } = 3;

        // --- UI CATEGORY 3: Adaptive Z-Score ---
        [Category("3. Adaptive Z-Score")] public int BaselineWindowSize { get; set; } = 1250;
        [Category("3. Adaptive Z-Score")] public bool FreezeBaselineDuringRipple { get; set; } = true;

        [Category("3. Adaptive Z-Score")]
        [Description("Scale applied to the z-scored value before the model. Set to match training " +
                     "(your previous pipeline used 0.5). 1.0 = pure z-score.")]
        public float ZScoreOutputScale { get; set; } = 1.0f;

        // --- UI CATEGORY 4: Thresholds ---
        [Category("4. Thresholds")] public float Threshold1_IgnoreBelow { get; set; } = 0.10f;
        [Category("4. Thresholds")] public float Threshold2_WeakEvidence { get; set; } = 0.50f;
        [Category("4. Thresholds")] public float Threshold3_StrongEvidence { get; set; } = 0.80f;
        [Category("4. Thresholds")] public float TargetEvidenceScore { get; set; } = 5.0f;

        // --- UI CATEGORY 5: Anti-Flicker ---
        [Category("5. Anti-Flicker")] public int FramesToWaitBeforeDecay { get; set; } = 5;
        [Category("5. Anti-Flicker")] public float ScoreDecayPerFrame { get; set; } = 1.0f;

        // --- UI CATEGORY 6: Hardware TTL ---
        [Category("6. Hardware TTL")] public int PostRippleRefractoryMs { get; set; } = 50;

        // --- INTERNAL STATE ---
        private readonly object _engineLock = new object();
        private bool _isInitialized = false;

        // Structural sizes captured at init (buffers are allocated once).
        private int _channels;
        private int _timePoints;
        private int _baselineWindow;

        // 1. Downsample State
        private int _decimationCounter = 0;

        // 2. Z-Score State (per-channel rolling rectangular window, double accumulators)
        private float[] _zHistory;       // [c * _baselineWindow + ringIndex]
        private double[] _zSum;
        private double[] _zSumSq;
        private int _zWriteIndex = 0;
        private int _zSamplesSeen = 0;

        // 3. ONNX State — _bufIn is a LINEAR, time-major sliding window [TimePoints, Channels]
        private InferenceSession _session;
        private OrtIoBinding _binding;
        private RunOptions _runOpts;
        private float[] _bufIn, _bufOut;
        private GCHandle _hIn, _hOut;
        private OrtValue _valIn, _valOut;
        private int _strideCounter = 0;

        // 4. FSM State
        private RippleState _fsmState = RippleState.NoRipple;
        private float _scoreTicks;
        private int _ticksInPossible;
        private int _ticksDipping;
        private int _eventCount;
        private bool _ttlHolding;
        private long _ttlHoldUntilMs;
        private static readonly Stopwatch Clock = Stopwatch.StartNew();

        private RippleOut _currentOutput;

        private void InitializeEngine()
        {
            if (_isInitialized) return;

            try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { }

            try
            {
                using (var process = System.Diagnostics.Process.GetCurrentProcess())
                    process.PriorityClass = System.Diagnostics.ProcessPriorityClass.RealTime;
                System.Threading.Thread.CurrentThread.Priority = System.Threading.ThreadPriority.Highest;
            }
            catch { }

            _channels = Channels;
            _timePoints = TimePoints;
            _baselineWindow = BaselineWindowSize;

            // 1. Z-Score buffers
            _zHistory = new float[_channels * _baselineWindow];
            _zSum = new double[_channels];
            _zSumSq = new double[_channels];

            // 2. ONNX linear window + output, pinned
            _bufIn = new float[_timePoints * _channels];
            _hIn = GCHandle.Alloc(_bufIn, GCHandleType.Pinned);
            _bufOut = new float[1]; // single probability output
            _hOut = GCHandle.Alloc(_bufOut, GCHandleType.Pinned);

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
            opts.AddSessionConfigEntry("session.execution_mode", "ORT_SEQUENTIAL");
            opts.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");

            if (UseCUDA) opts.AppendExecutionProvider_CUDA(DeviceId);

            _session = new InferenceSession(ModelPath, opts);
            _binding = _session.CreateIoBinding();
            _runOpts = new RunOptions();
            opts.Dispose();

            long[] inShape = new long[] { 1, _timePoints, _channels };
            long[] outShape = new long[] { 1, 1 };

            var memInfo = OrtMemoryInfo.DefaultInstance;
            _valIn = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufIn), inShape);
            _valOut = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufOut), outShape);

            _binding.BindInput(_session.InputMetadata.Keys.First(), _valIn);
            _binding.BindOutput(_session.OutputMetadata.Keys.First(), _valOut);

            for (int i = 0; i < 50; i++) _session.RunWithBinding(_runOpts, _binding);

            _currentOutput = new RippleOut { State = RippleState.NoRipple, TTL = false };
            _isInitialized = true;
            Clock.Restart();
        }

        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                lock (_engineLock)
                {
                    if (!_isInitialized) InitializeEngine();

                    int channels = _channels;
                    int timePoints = _timePoints;
                    int baseWin = _baselineWindow;
                    int decim = DecimationFactor < 1 ? 1 : DecimationFactor;
                    int stride = InferenceStride < 1 ? 1 : InferenceStride;
                    float zScale = ZScoreOutputScale;

                    int nCols = input.Cols;
                    int rowStep = input.Step;          // bytes per channel-major row
                    if (input.Rows < channels)
                        throw new InvalidOperationException(
                            $"Input has {input.Rows} rows but Channels={channels}. Feed a [Channels x N] " +
                            $"channel-major Mat (use SelectChannels upstream).");

                    int shiftBytes = (timePoints - 1) * channels * sizeof(float);

                    unsafe
                    {
                        byte* baseB = (byte*)input.Data.ToPointer();
                        float* win = (float*)_hIn.AddrOfPinnedObject().ToPointer();

                        // Walk every incoming sample (column) so this node can sit directly on the
                        // block-rate raw stream — no upstream Buffer(1) explosion, zero allocation.
                        for (int t = 0; t < nCols; t++)
                        {
                            // --- 1. DECIMATION (phase persists across blocks) ---
                            if (++_decimationCounter < decim) continue;
                            _decimationCounter = 0;

                            bool freeze = FreezeBaselineDuringRipple &&
                                          (_fsmState == RippleState.Ripple || _ttlHolding);
                            int n_t = _zSamplesSeen < 1 ? 1 : _zSamplesSeen;

                            // --- 2. Slide the linear time-major window left by one timepoint ---
                            //     (left shift => forward copy is safe for overlapping regions)
                            Buffer.MemoryCopy(win + channels, win, shiftBytes, shiftBytes);
                            float* newSlot = win + (timePoints - 1) * channels;

                            // --- 3. Adaptive rectangular Z-Score + ingest newest sample ---
                            for (int c = 0; c < channels; c++)
                            {
                                float newValue = *(float*)(baseB + c * rowStep + t * sizeof(float));
                                int histOff = c * baseWin + _zWriteIndex;

                                if (!freeze)
                                {
                                    float oldValue = (_zSamplesSeen < baseWin) ? 0f : _zHistory[histOff];
                                    _zSum[c]   += newValue - oldValue;
                                    _zSumSq[c] += (double)newValue * newValue - (double)oldValue * oldValue;
                                    _zHistory[histOff] = newValue;
                                }

                                double mu = _zSum[c] / n_t;
                                double variance = Math.Max(0.0, _zSumSq[c] / n_t - mu * mu);
                                double sigma = Math.Max(1e-10, Math.Sqrt(variance));
                                newSlot[c] = (float)(((newValue - mu) / sigma) * zScale);
                            }

                            if (!freeze)
                            {
                                _zWriteIndex = (_zWriteIndex + 1) % baseWin;
                                if (_zSamplesSeen < baseWin) _zSamplesSeen++;
                            }

                            // --- 4. Inference on a fully chronological window, on stride ---
                            if (++_strideCounter >= stride)
                            {
                                _strideCounter = 0;
                                if (_zSamplesSeen >= timePoints) // window holds real data
                                {
                                    _session.RunWithBinding(_runOpts, _binding);
                                    float prob = *((float*)_hOut.AddrOfPinnedObject().ToPointer());
                                    RunFSM(prob);
                                }
                            }
                        }
                    }

                    return _currentOutput;
                }
            });
        }

        private void RunFSM(float probability)
        {
            long now = Clock.ElapsedMilliseconds;

            // 1. Refractory Period
            if (_ttlHolding)
            {
                if (now >= _ttlHoldUntilMs)
                {
                    _ttlHolding = false;
                    _fsmState = RippleState.NoRipple;
                    _scoreTicks = 0;
                    _ticksInPossible = 0;
                    _ticksDipping = 0;
                }
                else
                {
                    UpdateOutput(probability, true);
                    return;
                }
            }

            // 2. FSM Logic
            switch (_fsmState)
            {
                case RippleState.NoRipple:
                    if (probability >= Threshold1_IgnoreBelow)
                    {
                        _fsmState = RippleState.Possible;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;
                        _ticksDipping = 0;
                    }
                    break;

                case RippleState.Possible:
                    if (probability >= Threshold1_IgnoreBelow)
                    {
                        _ticksDipping = 0;
                        _ticksInPossible++;

                        if (probability >= Threshold3_StrongEvidence) _scoreTicks += 2.0f;
                        else if (probability >= Threshold2_WeakEvidence) _scoreTicks += 1.0f;

                        if (_scoreTicks >= TargetEvidenceScore)
                        {
                            _fsmState = RippleState.Ripple;
                            _eventCount++;
                            _scoreTicks = 0;
                            _ticksInPossible = 0;
                            _ticksDipping = 0;

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
                                _scoreTicks = 0;
                                _ticksInPossible = 0;
                                _ticksDipping = 0;
                            }
                        }
                    }
                    break;

                case RippleState.Ripple:
                    if (probability < Threshold1_IgnoreBelow)
                    {
                        _fsmState = RippleState.NoRipple;
                        _scoreTicks = 0;
                        _ticksDipping = 0;
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
            lock (_engineLock)
            {
                if (_hIn.IsAllocated) _hIn.Free();
                if (_hOut.IsAllocated) _hOut.Free();
                _valIn?.Dispose();
                _valOut?.Dispose();
                _binding?.Dispose();
                _runOpts?.Dispose();
                _session?.Dispose();
                _valIn = null;
                _valOut = null;
                _binding = null;
                _runOpts = null;
                _session = null;
                _isInitialized = false;
            }
        }
    }
}
