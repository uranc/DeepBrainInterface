using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Drawing.Design; // Required for UITypeEditor
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
        // 1. CONFIGURATION
        // ==============================================================================
        [Category("Model")]
        [Description("Path to the ONNX model file.")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        [FileNameFilter("ONNX Models|*.onnx|All Files|*.*")]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("Model Dimensions")]
        public int TimePoints { get; set; } = 44;

        [Category("Model Dimensions")]
        public int Channels { get; set; } = 8;

        [Category("Model Dimensions")]
        [Description("Must be at least 2 for (Signal + Artifact) parallel inference.")]
        public int BatchSize { get; set; } = 2;

        [Category("Logic")]
        [Description("Threshold for Index 1 (Artifact) of the model output.")]
        public float ArtifactThreshold { get; set; } = 0.5f;

        [Category("Logic")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public RippleStateMachineMatBool StateMachine { get; set; } = new RippleStateMachineMatBool();

        [Category("Adaptive Stride")]
        public int KBelowGate { get; set; } = 5;

        [Category("Adaptive Stride")]
        public int KAtGate { get; set; } = 1;

        // ==============================================================================
        // 2. PROCESS
        // ==============================================================================

        // Input Tuple Structure: ((RawSignal, RawArtifactInput), ExternalGate)
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
        {
            // Observable.Using ensures 'engine.Dispose()' is called when workflow stops
            return Observable.Using(
                () => new AdaptiveEngine(ModelPath, BatchSize, Channels, TimePoints),
                (AdaptiveEngine engine) =>
                    source.Select(input =>
                    {
                        var mats = input.Item1;
                        var extGate = input.Item2;
                        Mat rawSig = mats.Item1;
                        Mat rawArt = mats.Item2;

                        // Safe execution
                        return engine.Execute(rawSig, rawArt, extGate, StateMachine, ArtifactThreshold, KBelowGate, KAtGate);
                    })
            );
        }

        // ==============================================================================
        // 3. ENGINE (Internal Resource Management)
        // ==============================================================================
        private class AdaptiveEngine : IDisposable
        {
            private InferenceSession _session;
            private OrtIoBinding _binding;
            private RunOptions _runOpts;

            // Pinned Memory Handles (Prevent GC moving memory during C++ calls)
            private GCHandle _hInput, _hOutput;
            private float[] _outBuffer;

            // Layout helpers
            private int _strideFloats;
            private int _totalInputBytes;
            private int _batchSize;

            // Stride State
            private int _strideCounter = 0;
            private int _currentK = 1;

            public AdaptiveEngine(string path, int batch, int channels, int time)
            {
                // 1. SAFETY CHECK
                if (batch < 2) throw new ArgumentException("BatchSize must be at least 2 (Index 0=Signal, Index 1=Artifact).");

                // 2. PROCESS PRIORITY (P-Core Optimization)
                // Safely requesting High priority ensures this thread runs on a Performance Core
                // without freezing the mouse (unlike RealTime).
                try
                {
                    using (var proc = System.Diagnostics.Process.GetCurrentProcess())
                    {
                        if (proc.PriorityClass != System.Diagnostics.ProcessPriorityClass.High)
                            proc.PriorityClass = System.Diagnostics.ProcessPriorityClass.High;
                    }
                }
                catch { /* Ignore permission errors */ }

                _batchSize = batch;
                _strideFloats = time * channels;
                _totalInputBytes = _strideFloats * sizeof(float);

                // 3. ONNX CONFIGURATION (Anti-Freeze)
                // Restricting threads to 1 ensures we consume only ONE P-Core.
                var opts = new SessionOptions
                {
                    IntraOpNumThreads = 1,
                    InterOpNumThreads = 1,
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                };

                try
                {
                    _session = new InferenceSession(path, opts);
                    _binding = _session.CreateIoBinding();
                    _runOpts = new RunOptions();

                    // 4. MEMORY PINNING
                    // Input: [Batch * Time * Channels]
                    float[] inBuffer = new float[batch * _strideFloats];
                    _hInput = GCHandle.Alloc(inBuffer, GCHandleType.Pinned);

                    // Output: [Batch * 1]
                    _outBuffer = new float[batch];
                    _hOutput = GCHandle.Alloc(_outBuffer, GCHandleType.Pinned);

                    var mem = OrtMemoryInfo.DefaultInstance;

                    using (var inOrt = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(inBuffer), new long[] { batch, time, channels }))
                    using (var outOrt = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(_outBuffer), new long[] { batch, 1 }))
                    {
                        _binding.BindInput(_session.InputMetadata.Keys.First(), inOrt);
                        _binding.BindOutput(_session.OutputMetadata.Keys.First(), outOrt);
                    }

                    // Warmup Run
                    _session.RunWithBinding(_runOpts, _binding);
                }
                catch (Exception ex)
                {
                    Dispose(); // Cleanup if half-initialized
                    throw new Exception($"Model Initialization Failed: {ex.Message}");
                }
                finally
                {
                    opts.Dispose();
                }
            }

            public RippleOut Execute(Mat sig, Mat art, bool extGate, RippleStateMachineMatBool fsm, float artThresh, int kBelow, int kAt)
            {
                // --- STEP 1: Adaptive Stride Logic ---
                if (_currentK > 1)
                {
                    _strideCounter++;
                    if (_strideCounter < _currentK)
                    {
                        // SKIPPING INFERENCE
                        return new RippleOut
                        {
                            State = RippleState.NoRipple,
                            StrideUsed = _currentK,
                            SignalData = sig
                        };
                    }
                    _strideCounter = 0;
                }

                // --- STEP 2: Memory Copy (Merge Inputs) ---
                unsafe
                {
                    float* dstBase = (float*)_hInput.AddrOfPinnedObject();

                    // A. Copy Signal to Batch Index 0
                    float* srcSig = (float*)sig.Data.ToPointer();
                    Buffer.MemoryCopy(srcSig, dstBase, _totalInputBytes, _totalInputBytes);

                    // B. Copy Artifact to Batch Index 1 (if it exists)
                    if (art != null && _batchSize > 1)
                    {
                        // Offset pointer by 1 batch stride
                        float* srcArt = (float*)art.Data.ToPointer();
                        Buffer.MemoryCopy(srcArt, dstBase + _strideFloats, _totalInputBytes, _totalInputBytes);
                    }
                }

                // --- STEP 3: Run Inference ---
                _session.RunWithBinding(_runOpts, _binding);

                // --- STEP 4: Read Outputs ---
                float sigProb = _outBuffer[0];
                float artProb = _outBuffer[1];

                // --- STEP 5: Update State Machine ---
                bool isArtifact = artProb > artThresh;
                bool finalGate = extGate && !isArtifact;

                var result = fsm.Update(sigProb, finalGate, sig);

                // --- STEP 6: Adjust Stride for NEXT frame ---
                _currentK = (result.State != RippleState.NoRipple) ? kAt : kBelow;

                // Finalize Result
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