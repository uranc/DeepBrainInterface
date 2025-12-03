using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms.Design;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Adaptive Ripple Detector: Stride Gate + CPU Inference + State Machine.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorAdaptive
    {
        // ==============================================================================
        // 1. MODEL PARAMETERS
        // ==============================================================================
        [Category("Model")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";

        // No Provider (Hardcoded CPU)
        // No NumThreads (Hardcoded 1)

        [Category("Model")]
        public int BatchSize { get; set; } = 1;

        [Category("Model")]
        public int TimePoints { get; set; } = 44;

        [Category("Model")]
        public int Channels { get; set; } = 8;

        // ==============================================================================
        // 2. GENERAL
        // ==============================================================================
        [Category("General"), DisplayName("Detection Enabled")]
        public bool DetectionEnabled { get; set; } = true;

        // ==============================================================================
        // 3. STRIDE K
        // ==============================================================================
        [Category("Stride K"), DisplayName("K below Gate (Relaxed)")]
        public int KBelowGate { get; set; } = 5;

        [Category("Stride K"), DisplayName("K at Gate (Focus)")]
        public int KAtGate { get; set; } = 1;

        // ==============================================================================
        // 4. THRESHOLDS (Flat list, synced internally)
        // ==============================================================================
        [Category("Thresholds"), DisplayName("1. Gate (arm)")]
        public float GateThreshold { get; set; } = 0.10f;

        [Category("Thresholds"), DisplayName("2. Enter (+0.5 per tick)")]
        public float EnterThreshold { get; set; } = 0.50f;

        [Category("Thresholds"), DisplayName("3. Confirm (+1.0 per tick)")]
        public float ConfirmThreshold { get; set; } = 0.80f;

        [Category("Thresholds"), DisplayName("4. Event Score (≥ triggers)")]
        public float EventScoreThreshold { get; set; } = 2.5f;

        // ==============================================================================
        // 5. TTL / TIMING
        // ==============================================================================
        [Category("TTL"), DisplayName("PostRipple Hold (ms)")]
        public int PostRippleMs { get; set; } = 50;

        [Category("TTL"), DisplayName("Trigger Delay (ms)")]
        public int TriggerDelayMs { get; set; } = 0;

        // ──────────────────────────────────────────────────────────────────
        // INTERNAL STATE
        // ──────────────────────────────────────────────────────────────────
        InferenceSession _session;
        OrtIoBinding _ioBinding;
        RunOptions _runOptions;

        OrtValue _inputOrtValue;
        OrtValue _outputOrtValue;
        float[] _inputBuffer;
        float[] _outputBuffer;

        int _strideCounter;
        int _currentK = 1;

        // Private Logic Engine
        RippleStateMachineMatBool _stateMachine = new RippleStateMachineMatBool();

        private void Initialise()
        {
            if (_session != null) return;
            if (!File.Exists(ModelPath)) throw new FileNotFoundException("Model not found", ModelPath);

            _inputBuffer = new float[BatchSize * TimePoints * Channels];

            var opts = new SessionOptions
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1
            };

            // Hardcoded CPU

            _session = new InferenceSession(ModelPath, opts);
            _runOptions = new RunOptions();

            _ioBinding = _session.CreateIoBinding();
            var inputName = _session.InputMetadata.Keys.First();
            var outputName = _session.OutputMetadata.Keys.First();
            var memInfo = OrtMemoryInfo.DefaultInstance;

            long[] inputShape = new long[] { BatchSize, TimePoints, Channels };
            _inputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(memInfo, _inputBuffer, inputShape);
            _ioBinding.BindInput(inputName, _inputOrtValue);

            var modelDims = _session.OutputMetadata[outputName].Dimensions;
            long[] outputShape = new long[modelDims.Length];
            long totalSize = 1;
            for (int i = 0; i < modelDims.Length; i++)
            {
                long dim = modelDims[i];
                if (dim <= 0)
                {
                    if (i == 0) dim = BatchSize;
                    else if (modelDims.Length == 3 && i == 1) dim = TimePoints;
                    else dim = 1;
                }
                if (modelDims.Length == 3 && i == 1 && dim == 1 && TimePoints > 1) dim = TimePoints;

                outputShape[i] = dim;
                totalSize *= dim;
            }

            _outputBuffer = new float[totalSize];
            _outputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(memInfo, _outputBuffer, outputShape);
            _ioBinding.BindOutput(outputName, _outputOrtValue);

            _session.RunWithBinding(_runOptions, _ioBinding);
            _currentK = KBelowGate;
        }

        // Overload 1: Mat Only
        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            return ProcessInternal(source.Select(m => Tuple.Create(m, true)));
        }

        // Overload 2: Mat + BNO Bool
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
        {
            return ProcessInternal(source);
        }

        private IObservable<RippleOut> ProcessInternal(IObservable<Tuple<Mat, bool>> source)
        {
            return source.Where(input =>
            {
                // 1. STRIDE GATE
                int k = _currentK;
                if (k <= 1) return true;

                _strideCounter++;
                if (_strideCounter >= k)
                {
                    _strideCounter = 0;
                    return true;
                }
                return false;
            })
            .Select(input =>
            {
                Mat mat = input.Item1;
                bool bnoOk = input.Item2;

                // 2. INFERENCE
                Initialise();
                float probability = RunFastInference(mat);

                // 3. UPDATE STATE MACHINE PARAMETERS
                // We push our flat properties to the internal logic engine every frame
                // (This is very cheap, just bool/float assignments)
                _stateMachine.DetectionEnabled = DetectionEnabled;
                _stateMachine.GateThreshold = GateThreshold;
                _stateMachine.EnterThreshold = EnterThreshold;
                _stateMachine.ConfirmThreshold = ConfirmThreshold;
                _stateMachine.EventScoreThreshold = EventScoreThreshold;
                _stateMachine.TriggerDelayMs = TriggerDelayMs;
                _stateMachine.PostRippleMs = PostRippleMs;

                // 4. RUN LOGIC
                RippleOut output = _stateMachine.Update(probability, bnoOk, mat);

                // 5. FEEDBACK
                if (!bnoOk)
                {
                    _currentK = KBelowGate; // Forced Relax
                }
                else if (output.State == RippleState.Possible)
                {
                    _currentK = KAtGate; // Focus
                }
                else
                {
                    _currentK = KBelowGate; // Relax
                }

                output.StrideUsed = _currentK;
                return output;
            });
        }

        private float RunFastInference(Mat mat)
        {
            unsafe
            {
                fixed (float* dstBase = _inputBuffer)
                {
                    float* src = (float*)mat.Data.ToPointer();
                    int len = TimePoints * Channels;
                    Buffer.MemoryCopy(src, dstBase, len * sizeof(float), len * sizeof(float));
                }
            }

            _session.RunWithBinding(_runOptions, _ioBinding);

            unsafe
            {
                fixed (float* src = _outputBuffer)
                {
                    int stride = _outputBuffer.Length / BatchSize;
                    return src[stride - 1];
                }
            }
        }

        ~RippleDetectorAdaptive()
        {
            _inputOrtValue?.Dispose(); _outputOrtValue?.Dispose(); _ioBinding?.Dispose();
            _runOptions?.Dispose(); _session?.Dispose();
        }
    }
}