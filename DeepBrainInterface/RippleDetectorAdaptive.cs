using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Drawing.Design;
using System.Windows.Forms.Design;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Adaptive Ripple Detector: Fixed Batch (1 or 2) + Stride Gate + Logic.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorAdaptive
    {
        // ==============================================================================
        // 1. PARAMETERS
        // ==============================================================================
        [Category("Model")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("Model")]
        [Description("Set strictly to 1 (Signal) or 2 (Signal + Artifact).")]
        public int BatchSize { get; set; } = 2;

        [Category("Model")]
        public int TimePoints { get; set; } = 44;

        [Category("Model")]
        public int Channels { get; set; } = 8;

        // ==============================================================================
        // 2. LOGIC & STRIDE
        // ==============================================================================
        [Category("Logic")]
        [Description("Embedded State Machine (Thresholds, Delays).")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public RippleStateMachineMatBool StateMachine { get; set; } = new RippleStateMachineMatBool();

        [Category("Stride"), DisplayName("K (Relaxed)")]
        [Description("Frames to skip when NO ripple is detected.")]
        public int KBelowGate { get; set; } = 5;

        [Category("Stride"), DisplayName("K (Active)")]
        [Description("Frames to skip when ripple is POSSIBLE (usually 1).")]
        public int KAtGate { get; set; } = 1;

        // ==============================================================================
        // 3. INTERNAL RESOURCES
        // ==============================================================================
        private InferenceSession _session;
        private OrtIoBinding _ioBinding;
        private RunOptions _runOptions;

        // Pinned Buffers
        private GCHandle _inputPin;
        private GCHandle _outputPin;
        private OrtValue _inputOrtValue;
        private OrtValue _outputOrtValue;

        // Buffers
        private float[] _outputBuffer;
        private int _batchStrideFloats;

        // Logic State
        private int _strideCounter;
        private int _currentK = 1;

        private void Initialise()
        {
            if (_session != null) return;
            if (!File.Exists(ModelPath)) throw new FileNotFoundException("Model not found", ModelPath);

            // A. SESSION CONFIG (Strict CPU)
            var opts = new SessionOptions
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                EnableCpuMemArena = true
            };
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");

            _session = new InferenceSession(ModelPath, opts);
            _ioBinding = _session.CreateIoBinding();
            _runOptions = new RunOptions();

            // B. SIZE CALCULATIONS
            _batchStrideFloats = TimePoints * Channels;
            int totalInputFloats = BatchSize * _batchStrideFloats;

            // C. PINNED ALLOCATION
            var inputData = new float[totalInputFloats];
            _inputPin = GCHandle.Alloc(inputData, GCHandleType.Pinned);

            _outputBuffer = new float[BatchSize];
            _outputPin = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);

            // D. BINDING
            var memInfo = OrtMemoryInfo.DefaultInstance;
            unsafe
            {
                _inputOrtValue = OrtValue.CreateTensorValueWithData(
                    memInfo, TensorElementType.Float,
                    new long[] { BatchSize, TimePoints, Channels },
                    _inputPin.AddrOfPinnedObject(),
                    totalInputFloats * sizeof(float)
                );

                _outputOrtValue = OrtValue.CreateTensorValueWithData(
                    memInfo, TensorElementType.Float,
                    new long[] { BatchSize, 1 },
                    _outputPin.AddrOfPinnedObject(),
                    _outputBuffer.Length * sizeof(float)
                );
            }

            _ioBinding.BindInput(_session.InputMetadata.Keys.First(), _inputOrtValue);
            _ioBinding.BindOutput(_session.OutputMetadata.Keys.First(), _outputOrtValue);

            // Warmup
            _session.RunWithBinding(_runOptions, _ioBinding);
        }

        // ==============================================================================
        // 4. PROCESSING LOGIC
        // ==============================================================================

        // --- BATCH 2 OVERLOADS ---

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return source.Where(_ => CheckStride()).Select(t => RunBatch2(t.Item1, t.Item2, true));
        }

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
        {
            return source.Where(_ => CheckStride()).Select(t => RunBatch2(t.Item1.Item1, t.Item1.Item2, t.Item2));
        }

        // --- BATCH 1 OVERLOADS ---

        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            return source.Where(_ => CheckStride()).Select(m => RunBatch1(m, true));
        }

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
        {
            return source.Where(_ => CheckStride()).Select(t => RunBatch1(t.Item1, t.Item2));
        }


        // ==============================================================================
        // 5. CORE EXECUTION
        // ==============================================================================

        private bool CheckStride()
        {
            // If K=1, always run. If K>1, skip (K-1) frames.
            if (_currentK <= 1) return true;

            _strideCounter++;
            if (_strideCounter >= _currentK)
            {
                _strideCounter = 0;
                return true;
            }
            return false;
        }

        private RippleOut RunBatch2(Mat signal, Mat artifact, bool bnoOk)
        {
            Initialise();
            if (BatchSize != 2) throw new InvalidOperationException("Adaptive Config is Batch 2, but received Batch 1.");

            unsafe
            {
                float* ptr = (float*)_inputPin.AddrOfPinnedObject();
                RobustTransposeCopy(signal, ptr);
                RobustTransposeCopy(artifact, ptr + _batchStrideFloats);
            }

            _session.RunWithBinding(_runOptions, _ioBinding);

            // Read Output Directly from Pinned Buffer
            float sigProb = _outputBuffer[0];
            float artProb = _outputBuffer[1];

            return UpdateState(sigProb, artProb, bnoOk, signal);
        }

        private RippleOut RunBatch1(Mat signal, bool bnoOk)
        {
            Initialise();
            if (BatchSize != 1) throw new InvalidOperationException("Adaptive Config is Batch 1, but received Batch 2.");

            unsafe
            {
                RobustTransposeCopy(signal, (float*)_inputPin.AddrOfPinnedObject());
            }

            _session.RunWithBinding(_runOptions, _ioBinding);

            // Read Output
            float sigProb = _outputBuffer[0];

            return UpdateState(sigProb, 0f, bnoOk, signal);
        }

        private RippleOut UpdateState(float sigP, float artP, bool bnoOk, Mat displayMat)
        {
            // Update State Machine
            var result = StateMachine.Update(sigP, artP, bnoOk, displayMat);

            // ADAPTIVE LOGIC:
            // If we are in a Ripple or Candidate state, we go FAST (K=1).
            // If we are in NoRipple state, we go SLOW (K=5 or user defined).
            if (result.State != RippleState.NoRipple)
                _currentK = KAtGate;
            else
                _currentK = KBelowGate;

            result.StrideUsed = _currentK;
            return result;
        }

        // ==============================================================================
        // 6. ROBUST MEMORY OPS
        // ==============================================================================

        private unsafe void RobustTransposeCopy(Mat src, float* dstBase)
        {
            if (src.Rows != Channels || src.Cols != TimePoints)
                throw new InvalidOperationException($"Input must be {Channels}x{TimePoints}. Got {src.Rows}x{src.Cols}");

            byte* srcRowPtr = (byte*)src.Data.ToPointer();
            int srcStep = src.Step;

            for (int c = 0; c < Channels; c++)
            {
                float* srcFloatRow = (float*)srcRowPtr;
                for (int t = 0; t < TimePoints; t++)
                {
                    dstBase[t * Channels + c] = srcFloatRow[t];
                }
                srcRowPtr += srcStep;
            }
        }

        public void Unload()
        {
            _inputOrtValue?.Dispose();
            _outputOrtValue?.Dispose();
            if (_inputPin.IsAllocated) _inputPin.Free();
            if (_outputPin.IsAllocated) _outputPin.Free();
            _ioBinding?.Dispose();
            _runOptions?.Dispose();
            _session?.Dispose();
            _session = null;
        }
    }
}