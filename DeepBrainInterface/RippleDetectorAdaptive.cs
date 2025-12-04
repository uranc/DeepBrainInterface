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
        // 1. MODEL PARAMS
        [Category("Model")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";
        [Category("Model")] public int BatchSize { get; set; } = 1;
        [Category("Model")] public int TimePoints { get; set; } = 44;
        [Category("Model")] public int Channels { get; set; } = 8;

        // 2. GENERAL
        [Category("General"), DisplayName("Detection Enabled")] public bool DetectionEnabled { get; set; } = true;

        // 3. STRIDE
        [Category("Stride K"), DisplayName("K below Gate (Relaxed)")] public int KBelowGate { get; set; } = 5;
        [Category("Stride K"), DisplayName("K at Gate (Focus)")] public int KAtGate { get; set; } = 1;

        // 4. LOGIC
        [Category("Logic")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public RippleStateMachineMatBool StateMachine { get; set; } = new RippleStateMachineMatBool();

        // STATE
        InferenceSession _session;
        OrtIoBinding _ioBinding;
        RunOptions _runOptions;
        OrtValue _inputOrtValue;
        OrtValue _outputOrtValue;
        float[] _inputBuffer;
        float[] _outputBuffer;
        int _strideCounter;
        int _currentK = 1;

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

            _session = new InferenceSession(ModelPath, opts);
            _runOptions = new RunOptions();
            _ioBinding = _session.CreateIoBinding();

            var memInfo = OrtMemoryInfo.DefaultInstance;
            long[] inputShape = new long[] { BatchSize, TimePoints, Channels };
            _inputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(memInfo, _inputBuffer, inputShape);
            _ioBinding.BindInput(_session.InputMetadata.Keys.First(), _inputOrtValue);

            var modelDims = _session.OutputMetadata[_session.OutputMetadata.Keys.First()].Dimensions;
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
            _ioBinding.BindOutput(_session.OutputMetadata.Keys.First(), _outputOrtValue);

            _session.RunWithBinding(_runOptions, _ioBinding);
            _currentK = KBelowGate;
        }

        // --- OVERLOADS (Convenience) ---

        // 1. Generic List (Best for 4+ inputs or dynamic arrays)
        public IObservable<RippleOut> Process(IObservable<IList<Mat>> source)
        {
            return ProcessInternal(source.Select(list => new InputPackage { Mats = list, BnoOk = true }));
        }

        // 2. Single Mat
        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            return ProcessInternal(source.Select(m => new InputPackage { Mats = new[] { m }, BnoOk = true }));
        }

        // 3. Tuple (2 Mats) - e.g. Zip output
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1, t.Item2 }, BnoOk = true }));
        }

        // 4. Tuple (3 Mats)
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, Mat, Mat>> source)
        {
            return ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1, t.Item2, t.Item3 }, BnoOk = true }));
        }

        // 5. Tuple (4 Mats)
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, Mat, Mat, Mat>> source)
        {
            return ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1, t.Item2, t.Item3, t.Item4 }, BnoOk = true }));
        }

        // 6. Tuple (2 Mats) + Bool (Signal + Artifact + BNO)
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
        {
            return ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1.Item1, t.Item1.Item2 }, BnoOk = t.Item2 }));
        }

        // 7. Tuple (4 Mats) + Bool (Signal + 3 Artifacts + BNO)
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat, Mat, Mat>, bool>> source)
        {
            return ProcessInternal(source.Select(t => new InputPackage
            {
                Mats = new[] { t.Item1.Item1, t.Item1.Item2, t.Item1.Item3, t.Item1.Item4 },
                BnoOk = t.Item2
            }));
        }

        // Helper Struct
        struct InputPackage { public IList<Mat> Mats; public bool BnoOk; }

        private IObservable<RippleOut> ProcessInternal(IObservable<InputPackage> source)
        {
            return source.Where(input =>
            {
                int k = _currentK;
                if (k <= 1) return true;
                _strideCounter++;
                if (_strideCounter >= k) { _strideCounter = 0; return true; }
                return false;
            })
            .Select(input =>
            {
                Initialise();

                // 1. INFERENCE (Handles any number of mats up to BatchSize)
                float[] results = RunFastInference(input.Mats);

                // 2. EXTRACT SIGNAL & ARTIFACT
                // Assume Batch 0 = Signal
                float signalProb = (results.Length > 0) ? results[0] : 0f;

                // Assume Batch 1 = Main Artifact (or Max of remaining batches)
                float artifactProb = 0f;
                if (results.Length > 1)
                {
                    // If 4 batches, you might want the MAX of all artifact channels?
                    // Here we just take Batch 1, or check all subsequent batches
                    for (int i = 1; i < results.Length; i++)
                    {
                        artifactProb = Math.Max(artifactProb, results[i]);
                    }
                }

                StateMachine.DetectionEnabled = DetectionEnabled;

                // 3. LOGIC
                // Use Signal from Batch 0 for snapshot
                Mat signalSnapshot = (input.Mats.Count > 0) ? input.Mats[0] : null;
                RippleOut output = StateMachine.Update(signalProb, artifactProb, input.BnoOk, signalSnapshot);

                // 4. FEEDBACK
                bool artifactOk = artifactProb < StateMachine.ArtifactThreshold;
                bool gatesOpen = input.BnoOk && artifactOk;

                if (!gatesOpen) _currentK = KBelowGate;
                else if (output.State == RippleState.Possible) _currentK = KAtGate;
                else _currentK = KBelowGate;

                output.StrideUsed = _currentK;
                return output;
            });
        }

        private float[] RunFastInference(IList<Mat> mats)
        {
            // COPY INPUTS
            unsafe
            {
                fixed (float* dstBase = _inputBuffer)
                {
                    int singleBatchLen = TimePoints * Channels;

                    // Iterate up to BatchSize or InputCount, whichever is smaller
                    int count = Math.Min(BatchSize, mats.Count);

                    for (int b = 0; b < count; b++)
                    {
                        float* src = (float*)mats[b].Data.ToPointer();
                        float* dstOffset = dstBase + (b * singleBatchLen);
                        Buffer.MemoryCopy(src, dstOffset, singleBatchLen * sizeof(float), singleBatchLen * sizeof(float));
                    }
                }
            }

            _session.RunWithBinding(_runOptions, _ioBinding);

            // EXTRACT RESULTS
            float[] results = new float[BatchSize];
            unsafe
            {
                fixed (float* src = _outputBuffer)
                {
                    int stride = _outputBuffer.Length / BatchSize;
                    for (int b = 0; b < BatchSize; b++)
                    {
                        // Grab last timepoint of each batch
                        results[b] = src[(b * stride) + (stride - 1)];
                    }
                }
            }
            return results;
        }

        ~RippleDetectorAdaptive()
        {
            _inputOrtValue?.Dispose(); _outputOrtValue?.Dispose(); _ioBinding?.Dispose();
            _runOptions?.Dispose(); _session?.Dispose();
        }
    }
}