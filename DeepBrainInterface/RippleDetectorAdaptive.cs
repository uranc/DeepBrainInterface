using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
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
        // INTERNAL RESOURCES
        // ==============================================================================
        InferenceSession _session;
        OrtIoBinding _ioBinding;
        RunOptions _runOptions;
        OrtMemoryInfo _memInfo;

        // Fixed Pinned Buffers (Allocated Once)
        float[] _inputBuffer;
        GCHandle _inputPin;
        OrtValue _inputOrtValue;

        float[] _outputBuffer;
        GCHandle _outputPin;
        OrtValue _outputOrtValue;

        // Logic State
        int _strideCounter;
        int _currentK = 1;

        struct InputPackage { public Mat[] Mats; public bool BnoOk; }

        private void Initialise()
        {
            if (_session != null) return;
            if (!File.Exists(ModelPath)) throw new FileNotFoundException("Model not found", ModelPath);

            // A. CONFIGURE SESSION
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

            _session = new InferenceSession(ModelPath, opts);
            _memInfo = OrtMemoryInfo.DefaultInstance;
            _ioBinding = _session.CreateIoBinding();
            _runOptions = new RunOptions();

            // B. STRICT MEMORY ALLOCATION (Based on BatchSize)
            int inputLen = BatchSize * TimePoints * Channels;
            _inputBuffer = new float[inputLen];
            _inputPin = GCHandle.Alloc(_inputBuffer, GCHandleType.Pinned);

            int outputLen = BatchSize; // [Batch, 1]
            _outputBuffer = new float[outputLen];
            _outputPin = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);

            // C. BINDING SETUP
            var inputName = _session.InputMetadata.Keys.First();
            var outputName = _session.OutputMetadata.Keys.First();

            long[] inShape = new long[] { BatchSize, TimePoints, Channels };
            long[] outShape = new long[] { BatchSize, 1 };

            unsafe
            {
                _inputOrtValue = OrtValue.CreateTensorValueWithData(
                    _memInfo,
                    TensorElementType.Float,
                    inShape,
                    _inputPin.AddrOfPinnedObject(),
                    inputLen * sizeof(float)
                );

                _outputOrtValue = OrtValue.CreateTensorValueWithData(
                    _memInfo,
                    TensorElementType.Float,
                    outShape,
                    _outputPin.AddrOfPinnedObject(),
                    outputLen * sizeof(float)
                );
            }

            _ioBinding.BindInput(inputName, _inputOrtValue);
            _ioBinding.BindOutput(outputName, _outputOrtValue);

            // Warmup
            _session.RunWithBinding(_runOptions, _ioBinding);
        }

        private IObservable<RippleOut> ProcessInternal(IObservable<InputPackage> source)
        {
            return source.Where(input =>
            {
                // STRIDE GATE
                if (_currentK <= 1) return true;
                _strideCounter++;
                if (_strideCounter >= _currentK)
                {
                    _strideCounter = 0;
                    return true;
                }
                return false;
            })
            .Select(input =>
            {
                Initialise();

                // Safety: Ensure we received exactly what we configured
                if (input.Mats.Length != BatchSize)
                    throw new InvalidOperationException($"Input mismatch: Received {input.Mats.Length} mats, expected BatchSize {BatchSize}");

                // 1. FAST COPY (Mat -> Pinned Buffer)
                unsafe
                {
                    float* dstBase = (float*)_inputPin.AddrOfPinnedObject().ToPointer();
                    int stride = TimePoints * Channels;
                    long bytesPerMat = stride * sizeof(float);

                    for (int i = 0; i < BatchSize; i++)
                    {
                        float* src = (float*)input.Mats[i].Data.ToPointer();
                        float* dst = dstBase + (i * stride);

                        // Layout Check
                        if (input.Mats[i].Rows == TimePoints && input.Mats[i].Cols == Channels)
                        {
                            Buffer.MemoryCopy(src, dst, bytesPerMat, bytesPerMat);
                        }
                        else if (input.Mats[i].Rows == Channels && input.Mats[i].Cols == TimePoints)
                        {
                            // Transpose
                            for (int c = 0; c < Channels; c++)
                            {
                                int cOff = c * TimePoints;
                                for (int t = 0; t < TimePoints; t++)
                                {
                                    dst[t * Channels + c] = src[cOff + t];
                                }
                            }
                        }
                        else
                        {
                            Buffer.MemoryCopy(src, dst, bytesPerMat, bytesPerMat);
                        }
                    }
                }

                // 2. INFERENCE
                _session.RunWithBinding(_runOptions, _ioBinding);

                // 3. READ OUTPUTS (Strict Logic)
                float signalProb = _outputBuffer[0];
                float artifactProb = 0f;

                if (BatchSize >= 2)
                {
                    artifactProb = _outputBuffer[1];
                }

                // 4. LOGIC UPDATE
                RippleOut output = StateMachine.Update(signalProb, artifactProb, input.BnoOk, input.Mats[0]);

                // 5. ADAPT STRIDE
                if (output.State != RippleState.NoRipple) _currentK = KAtGate;
                else _currentK = KBelowGate;

                output.StrideUsed = _currentK;
                return output;
            });
        }

        // --- OVERLOADS ---

        // BatchSize = 1
        public IObservable<RippleOut> Process(IObservable<Mat> source)
            => ProcessInternal(source.Select(m => new InputPackage { Mats = new[] { m }, BnoOk = true }));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
            => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1 }, BnoOk = t.Item2 }));

        // BatchSize = 2
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, Mat>> source)
            => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1, t.Item2 }, BnoOk = true }));

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
            => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1.Item1, t.Item1.Item2 }, BnoOk = t.Item2 }));


        public void Unload()
        {
            _inputOrtValue?.Dispose(); _outputOrtValue?.Dispose();
            if (_inputPin.IsAllocated) _inputPin.Free();
            if (_outputPin.IsAllocated) _outputPin.Free();
            _ioBinding?.Dispose(); _runOptions?.Dispose(); _session?.Dispose();
        }
    }
}