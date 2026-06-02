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
    //public enum RippleState { NoRipple, Possible, Ripple }


    [Combinator]
    [Description("God Node: Downsampling -> Adaptive Z-Score -> ONNX -> Leaky Bucket FSM.")]
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
        [Description("Downsample factor. (e.g., 12 turns 30kHz into 2.5kHz)")]
        public int DecimationFactor { get; set; } = 12;

        [Category("2. Signal Timing")]
        [Description("How many 2.5kHz samples to wait before running inference again. (e.g., 2)")]
        public int InferenceStride { get; set; } = 2;

        // --- UI CATEGORY 3: Adaptive Z-Score ---
        [Category("3. Adaptive Z-Score")] public int BaselineWindowSize { get; set; } = 1250;
        [Category("3. Adaptive Z-Score")] public bool FreezeBaselineDuringRipple { get; set; } = true;

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

        // 1. Downsample State
        private int _decimationCounter = 0;

        // 2. Z-Score State
        private float[] _zHistory;
        private double[] _zSum;
        private double[] _zSumSq;
        private int _zWriteIndex = 0;
        private int _zSamplesSeen = 0;

        // 3. ONNX State
        private InferenceSession _session;
        private OrtIoBinding _binding;
        private RunOptions _runOpts;
        private float[] _bufIn, _bufOut;
        private GCHandle _hIn, _hOut;
        private OrtValue _valIn, _valOut;
        private int _onnxWriteIndex = 0;
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

            // 1. Init Z-Score
            _zHistory = new float[Channels * BaselineWindowSize];
            _zSum = new double[Channels];
            _zSumSq = new double[Channels];

            // 2. Init ONNX Ring Buffer & Memory
            int inSize = TimePoints * Channels;
            _bufIn = new float[inSize];
            _hIn = GCHandle.Alloc(_bufIn, GCHandleType.Pinned);

            _bufOut = new float[1]; // Assuming single probability float output
            _hOut = GCHandle.Alloc(_bufOut, GCHandleType.Pinned);

            var opts = new SessionOptions
            {
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                EnableCpuMemArena = true,
                EnableMemoryPattern = true
            };
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            opts.AddSessionConfigEntry("session.execution_mode", "ORT_SEQUENTIAL");

            if (UseCUDA) opts.AppendExecutionProvider_CUDA(DeviceId);

            _session = new InferenceSession(ModelPath, opts);
            _binding = _session.CreateIoBinding();
            _runOpts = new RunOptions();
            opts.Dispose();

            long[] inShape = new long[] { 1, TimePoints, Channels };
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

        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                lock (_engineLock)
                {
                    if (!_isInitialized) InitializeEngine();

                    // --- 1. DECIMATION ---
                    _decimationCounter++;
                    if (_decimationCounter < DecimationFactor)
                    {
                        return _currentOutput; // Yield previous state instantly
                    }
                    _decimationCounter = 0;

                    // --- 2. ADAPTIVE Z-SCORE ---
                    bool freezeZScore = FreezeBaselineDuringRipple && (_fsmState == RippleState.Ripple || _ttlHolding);
                    int zCount = Math.Min(_zSamplesSeen, BaselineWindowSize);

                    unsafe
                    {
                        float* src = (float*)input.Data.ToPointer();
                        float* onnxBufferBase = (float*)_hIn.AddrOfPinnedObject().ToPointer();

                        for (int c = 0; c < Channels; c++)
                        {
                            float newValue = src[c];
                            int historyOffset = (c * BaselineWindowSize) + _zWriteIndex;

                            // Adaptive Update
                            if (!freezeZScore)
                            {
                                float oldValue = (_zSamplesSeen < BaselineWindowSize) ? 0f : _zHistory[historyOffset];
                                _zSum[c] -= oldValue;
                                _zSumSq[c] -= (double)oldValue * oldValue;

                                // Anti-drift block
                                if (_zWriteIndex == 0 && _zSamplesSeen >= BaselineWindowSize)
                                {
                                    double freshSum = 0, freshSumSq = 0;
                                    int baseIdx = c * BaselineWindowSize;
                                    for (int j = 0; j < BaselineWindowSize; j++)
                                    {
                                        float val = _zHistory[baseIdx + j];
                                        freshSum += val;
                                        freshSumSq += (double)val * val;
                                    }
                                    _zSum[c] = freshSum;
                                    _zSumSq[c] = freshSumSq;
                                }

                                _zHistory[historyOffset] = newValue;
                                _zSum[c] += newValue;
                                _zSumSq[c] += (double)newValue * newValue;
                            }

                            // Normalization
                            int n_t = Math.Max(1, zCount);
                            double mu = _zSum[c] / n_t;
                            double avgSq = _zSumSq[c] / n_t;
                            double variance = Math.Max(0, avgSq - (mu * mu));
                            double sigma = Math.Max(1e-10, Math.Sqrt(variance));
                            float zScoredValue = (float)((newValue - mu) / sigma);

                            // --- 3. ONNX RING BUFFER INGESTION ---
                            // Write directly to the ONNX pinned memory at the current timepoint
                            int targetIndex = (_onnxWriteIndex * Channels) + c;
                            onnxBufferBase[targetIndex] = zScoredValue;
                        }

                        if (!freezeZScore)
                        {
                            _zWriteIndex = (_zWriteIndex + 1) % BaselineWindowSize;
                            _zSamplesSeen++;
                        }

                        _onnxWriteIndex = (_onnxWriteIndex + 1) % TimePoints;
                    }

                    // --- 4. INFERENCE STRIDE TRIGGER ---
                    _strideCounter++;
                    if (_strideCounter >= InferenceStride)
                    {
                        _strideCounter = 0;

                        // We do NOT need to reshape the buffer because ONNX doesn't care where "time zero" is
                        // as long as the neural network is trained on sliding windows. 
                        // If it strictly requires time-chronological ordering, we would shift it here.

                        _session.RunWithBinding(_runOpts, _binding);

                        unsafe
                        {
                            float prob = *((float*)_hOut.AddrOfPinnedObject().ToPointer());
                            RunFSM(prob);
                        }
                    }

                    return _currentOutput;
                }
            });
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
                _isInitialized = false;
            }
        }
    }
}