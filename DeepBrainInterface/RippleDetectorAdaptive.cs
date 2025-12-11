using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Adaptive Detector. Runs Inference -> Checks Artifact (Index 1) -> Updates FSM -> Adjusts Stride.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorAdaptive
    {
        // ==============================================================================
        // CONFIGURATION
        // ==============================================================================
        [Category("Model")]
        [Description("Path to the ONNX model file.")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        [FileNameFilter("ONNX Models|*.onnx|All Files|*.*")]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("Model Dimensions")] public int TimePoints { get; set; } = 44;
        [Category("Model Dimensions")] public int Channels { get; set; } = 8;
        [Category("Model Dimensions")] public int BatchSize { get; set; } = 2;

        [Category("Logic")]
        [Description("Threshold for Index 1 (Artifact). If exceeded, Gate closes.")]
        public float ArtifactThreshold { get; set; } = 0.5f;

        [Category("Logic")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public RippleStateMachineMatBool StateMachine { get; set; } = new RippleStateMachineMatBool();

        [Category("Adaptive Stride")] public int KBelowGate { get; set; } = 5;
        [Category("Adaptive Stride")] public int KAtGate { get; set; } = 1;

        // ==============================================================================
        // PROCESS
        // ==============================================================================
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
        {
            return Observable.Using(
                () => new AdaptiveEngine(ModelPath, BatchSize, Channels, TimePoints, StateMachine),
                (AdaptiveEngine engine) =>
                    source.Select(input =>
                    {
                        var mats = input.Item1;
                        var extGate = input.Item2;
                        Mat rawSig = mats.Item1;
                        Mat rawArt = mats.Item2;
                        return engine.Execute(rawSig, rawArt, extGate, ArtifactThreshold, KBelowGate, KAtGate);
                    })
            );
        }

        // ==============================================================================
        // ENGINE
        // ==============================================================================
        private class AdaptiveEngine : IDisposable
        {
            private InferenceSession _session;
            private OrtIoBinding _binding;
            private RunOptions _runOpts;

            // Persistent State
            private RippleStateMachineMatBool _fsm;
            private RippleOut _lastResult;

            // Resources
            private GCHandle _hInput, _hOutput;
            private float[] _outBuffer;

            // Dimensions
            private int _strideFloats;
            private long _strideBytes; // Using long for MemoryCopy size
            private int _batchSize;

            // Adaptive State
            private int _strideCounter = 0;
            private int _currentK = 1;

            public AdaptiveEngine(string path, int batch, int channels, int time, RippleStateMachineMatBool fsm)
            {
                _fsm = fsm;

                // Init cache to defaults
                _lastResult = new RippleOut
                {
                    State = RippleState.NoRipple,
                    TTL = false,
                    EventCount = 0,
                    Score = 0f,
                    Probability = 0f,
                    ArtifactProbability = 0f,
                    SignalData = null,
                    StrideUsed = 1
                };

                // P-Core Optimization
                try
                {
                    using (var proc = System.Diagnostics.Process.GetCurrentProcess())
                    {
                        if (proc.PriorityClass != System.Diagnostics.ProcessPriorityClass.High)
                            proc.PriorityClass = System.Diagnostics.ProcessPriorityClass.High;
                    }
                }
                catch { }

                if (batch < 2) throw new ArgumentException("BatchSize must be >= 2.");

                _batchSize = batch;
                _strideFloats = time * channels;
                _strideBytes = _strideFloats * sizeof(float);

                var opts = new SessionOptions
                {
                    IntraOpNumThreads = 1,
                    InterOpNumThreads = 1,
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
                };

                try
                {
                    _session = new InferenceSession(path, opts);
                    _binding = _session.CreateIoBinding();
                    _runOpts = new RunOptions();

                    // Alloc Buffers
                    float[] inBuffer = new float[batch * _strideFloats];
                    _hInput = GCHandle.Alloc(inBuffer, GCHandleType.Pinned);
                    _outBuffer = new float[batch];
                    _hOutput = GCHandle.Alloc(_outBuffer, GCHandleType.Pinned);

                    var mem = OrtMemoryInfo.DefaultInstance;
                    using (var inOrt = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(inBuffer), new long[] { batch, time, channels }))
                    using (var outOrt = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(_outBuffer), new long[] { batch, 1 }))
                    {
                        _binding.BindInput(_session.InputMetadata.Keys.First(), inOrt);
                        _binding.BindOutput(_session.OutputMetadata.Keys.First(), outOrt);
                    }
                    _session.RunWithBinding(_runOpts, _binding);
                }
                catch (Exception ex)
                {
                    Dispose();
                    throw new Exception($"Model Init Failed: {ex.Message}");
                }
                finally { opts.Dispose(); }
            }

            public RippleOut Execute(Mat sig, Mat art, bool extGate, float artThresh, int kBelow, int kAt)
            {
                // =========================================================
                // 1. ADAPTIVE STRIDE (SKIPPED FRAME LOGIC)
                // =========================================================
                if (_currentK > 1)
                {
                    _strideCounter++;
                    if (_strideCounter < _currentK)
                    {
                        // Explicitly check and copy EVERY property from _lastResult
                        return new RippleOut
                        {
                            // 1. State (Must persist)
                            State = _lastResult.State,

                            // 2. TTL (Must persist to hold High)
                            TTL = _lastResult.TTL,

                            // 3. Score (Must persist to avoid drop to 0)
                            Score = _lastResult.Score,

                            // 4. Event Count (Must persist)
                            EventCount = _lastResult.EventCount,

                            // 5. Probability (Must persist)
                            Probability = _lastResult.Probability,

                            // 6. Signal Data (Persist snapshot if exists)
                            SignalData = _lastResult.SignalData,

                            // 7. Artifact Prob (Must persist)
                            ArtifactProbability = _lastResult.ArtifactProbability,

                            // 8. Stride Used (Updated)
                            StrideUsed = _currentK
                        };
                    }
                    _strideCounter = 0;
                }

                // =========================================================
                // 2. MEMORY COPY (Reverted to standard Buffer.MemoryCopy)
                // =========================================================
                unsafe
                {
                    float* dstBase = (float*)_hInput.AddrOfPinnedObject();

                    // Copy Signal (Batch 0)
                    if (sig != null)
                    {
                        float* srcSig = (float*)sig.Data.ToPointer();
                        Buffer.MemoryCopy(srcSig, dstBase, _strideBytes, _strideBytes);
                    }

                    // Copy Artifact (Batch 1)
                    if (art != null && _batchSize > 1)
                    {
                        float* srcArt = (float*)art.Data.ToPointer();
                        // Offset destination by one full sample stride
                        Buffer.MemoryCopy(srcArt, dstBase + _strideFloats, _strideBytes, _strideBytes);
                    }
                }

                // =========================================================
                // 3. INFERENCE
                // =========================================================
                _session.RunWithBinding(_runOpts, _binding);

                float sigProb = _outBuffer[0];
                float artProb = _outBuffer[1];

                // =========================================================
                // 4. LOGIC & FSM
                // =========================================================
                bool isArtifact = artProb > artThresh;
                bool finalGate = extGate && !isArtifact;

                // Update persistent FSM
                var result = _fsm.Update(sigProb, finalGate, sig);

                // =========================================================
                // 5. CACHE UPDATE
                // =========================================================
                _lastResult = result;
                _currentK = (result.State != RippleState.NoRipple) ? kAt : kBelow;

                // Inject metadata
                result.StrideUsed = _currentK;
                result.ArtifactProbability = artProb;

                return result;
            }

            public void Dispose()
            {
                _session?.Dispose();
                _binding?.Dispose();
                _runOpts?.Dispose();
                if (_hInput.IsAllocated) _hInput.Free();
                if (_hOutput.IsAllocated) _hOutput.Free();
            }
        }
    }
}