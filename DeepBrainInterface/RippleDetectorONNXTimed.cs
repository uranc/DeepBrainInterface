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
using System.Reflection;
using System.Windows.Forms.Design;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("ONNX ripple detector with Δt-run / Δt-call timing. Uses fixed batch size for static ONNX input dimensions.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorONNXTimed
    {
        /* ───── user parameters ─────────────────────────────────────────── */
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } =
            @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";

        public OnnxProvider Provider { get; set; } = OnnxProvider.Cpu;

        [Description("Batch size must match the ONNX model's fixed first dimension.")]
        public int BatchSize { get; set; } = 1;

        public int ExpectedTimepoints { get; set; } = 92;
        public int ExpectedChannels { get; set; } = 8;
        /* ───────────────────────────────────────────────────────────────── */

        InferenceSession _session;
        string _inputName, _outputName;

        static readonly Stopwatch swRun = new Stopwatch();
        static readonly Stopwatch swLoop = Stopwatch.StartNew();
        static long prevTicks;
        const int kPrint = 1000;
        int nAccum;
        double accRun, accCall;

        void Initialise()
        {
            if (_session != null) return;

            var opts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = 6,
                InterOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_PARALLEL
            };
            opts.AddSessionConfigEntry("session.use_dnnl", "1");
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

            try
            {
                if (Provider == OnnxProvider.Cuda) opts.AppendExecutionProvider_CUDA(0);
                else if (Provider == OnnxProvider.TensorRT)
                {
                    opts.AppendExecutionProvider_CUDA(0);
                    opts.AppendExecutionProvider_Tensorrt();
                }
            }
            catch
            {
                Provider = OnnxProvider.Cpu;
            }

            if (!File.Exists(ModelPath))
                throw new FileNotFoundException("ONNX model not found", ModelPath);
            _session = new InferenceSession(ModelPath, opts);

            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();

            // Inspect the model's input shape: dims[0] is the model's batch dimension (static if >0, dynamic if -1)
            var dims = _session.InputMetadata[_inputName].Dimensions;
            // dims[0] must match BatchSize exactly
            // Enforce that the model uses a static batch size matching our parameter
            if (dims[0] <= 0)
                throw new InvalidOperationException(
                    $"Model declares a dynamic batch dimension (dims[0] = {dims[0]}). " +
                    "Static batch size is required for performance.");
            if (dims[0] != BatchSize)
                throw new InvalidOperationException(
                    $"Model expects static batch size={dims[0]}, but BatchSize is set to {BatchSize}.");
        }

        DenseTensor<float> ToTensor(IList<Mat> batch)
        {
            if (batch.Count != BatchSize)
                throw new ArgumentException(
                    $"Expected exactly {BatchSize} Mats in batch, but got {batch.Count}.");

            var dims = new[] { BatchSize, ExpectedTimepoints, ExpectedChannels };
            var buf = new float[BatchSize * ExpectedTimepoints * ExpectedChannels];
            int idx = 0;

            for (int b = 0; b < BatchSize; b++)
            {
                unsafe
                {
                    float* src = (float*)batch[b].Data.ToPointer();
                    for (int ch = 0; ch < ExpectedChannels; ch++)
                        for (int t = 0; t < ExpectedTimepoints; t++)
                            buf[idx++] = src[ch * ExpectedTimepoints + t];
                }
            }
            return new DenseTensor<float>(buf, dims);
        }

        // Single-stream input: wrap into a one-element batch if BatchSize==1
        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            if (BatchSize != 1) throw new InvalidOperationException(
                "BatchSize > 1 requires zipped input of IObservable<IList<Mat>>.");

            return source.Select(mat => ProcessBatch(new[] { mat }));
        }

        // Zipped input: receives exactly BatchSize Mats per emission
        public IObservable<Mat> Process(IObservable<IList<Mat>> source)
        {
            return source.Select(batch => ProcessBatch(batch));
        }

        // Overload for two-stream Zip: Tuple<Mat,Mat>
        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return source.Select(tuple => ProcessBatch(new[] { tuple.Item1, tuple.Item2 }));
        }

        private Mat ProcessBatch(IList<Mat> batch)
        {
            Initialise();
            var tensor = ToTensor(batch);
            var named = NamedOnnxValue.CreateFromTensor(_inputName, tensor);

            // inference
            swRun.Restart();
            var results = _session.Run(new[] { named });
            swRun.Stop();

            var outTensor = results.First().AsTensor<float>();
            results.Dispose();

            // timing
            long now = swLoop.ElapsedTicks;
            long dt = now - prevTicks;
            prevTicks = now;
            accRun += swRun.ElapsedTicks * 1e6 / Stopwatch.Frequency;
            accCall += dt * 1e6 / Stopwatch.Frequency;
            if (++nAccum == kPrint)
            {
                Console.WriteLine($"ONNX ⟨Δt-run⟩={accRun / kPrint:F2} µs   ⟨Δt-call⟩={accCall / kPrint:F2} µs");
                nAccum = 0; accRun = 0; accCall = 0;
            }

            // pack output into one multi-channel Mat
            int outLen = (int)(outTensor.Length / BatchSize);
            var flat = outTensor.ToArray();

            var outMat = new Mat(1, outLen, Depth.F32, BatchSize);
            unsafe
            {
                float* dst = (float*)outMat.Data.ToPointer();
                for (int t = 0; t < outLen; t++)
                    for (int ch = 0; ch < BatchSize; ch++)
                        dst[t * BatchSize + ch] = flat[ch * outLen + t];
            }
            return outMat;
        }
    }
}
