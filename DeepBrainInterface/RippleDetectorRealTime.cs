using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms.Design;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Ultra-low-latency ONNX inference supporting batch sizes 1 or 2 (1×T×C or 2×T×C → multi-channel 1×1 Mat)")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorRealTime
    {
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; }
            = @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";

        [Description("ONNX Runtime threads (2-6 recommended)")]
        public int NumThreads { get; set; } = 5;

        [Description("Batch size: 1 for single-buffer, 2 for zipped input")]
        public int BatchSize { get; set; } = 1;

        [Description("Time points per sample")]
        public int TimePoints { get; set; } = 92;

        [Description("Channels per sample")]
        public int Channels { get; set; } = 8;

        InferenceSession session;
        string inputName;
        float[] buffer;
        GCHandle bufferPin;
        readonly List<NamedOnnxValue> inputs = new List<NamedOnnxValue>(1);

        void Initialise()
        {
            if (session != null) return;
            // allocate pinned buffer for BatchSize*T*C
            buffer = new float[BatchSize * TimePoints * Channels];
            bufferPin = GCHandle.Alloc(buffer, GCHandleType.Pinned);

            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = NumThreads,
                InterOpNumThreads = 1,
                EnableCpuMemArena = true
            };
            options.AddSessionConfigEntry("session.enable_mem_pattern", "1");
            options.AddSessionConfigEntry("session.execution_mode", "ORT_SEQUENTIAL");
            options.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
            options.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

            session = new InferenceSession(ModelPath, options);
            inputName = session.InputMetadata.Keys.First();

            // Warm-up
            var warmup = new DenseTensor<float>(buffer, new[] { BatchSize, TimePoints, Channels });
            using (var _ = session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, warmup) })) { }
        }

        DenseTensor<float> BuildTensor(params Mat[] mats)
        {
            int stride = TimePoints * Channels;
            unsafe
            {
                var dstBase = (float*)bufferPin.AddrOfPinnedObject().ToPointer();
                for (int i = 0; i < mats.Length; i++)
                {
                    float* src = (float*)mats[i].Data.ToPointer();
                    float* dst = dstBase + i * stride;
                    if (mats[i].Rows == TimePoints && mats[i].Cols == Channels)
                    {
                        Buffer.MemoryCopy(src, dst, stride * sizeof(float), stride * sizeof(float));
                    }
                    else if (mats[i].Rows == Channels && mats[i].Cols == TimePoints)
                    {
                        for (int c = 0; c < Channels; c++)
                            for (int t = 0; t < TimePoints; t++)
                                dst[t * Channels + c] = src[c * TimePoints + t];
                    }
                    else throw new ArgumentException($"Unexpected Mat shape {mats[i].Rows}×{mats[i].Cols}");
                }
            }
            return new DenseTensor<float>(buffer, new[] { BatchSize, TimePoints, Channels });
        }

        // Single-buffer path
        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            if (BatchSize != 1)
                throw new InvalidOperationException("BatchSize must be 1 for single-buffer input.");
            return source.Select(m => RunInference(m));
        }

        // Dual-buffer zipped path
        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            if (BatchSize != 2)
                throw new InvalidOperationException("BatchSize must be 2 for Tuple<Mat,Mat> input.");
            return source.Select(t => RunInference(t.Item1, t.Item2));
        }

        private Mat RunInference(params Mat[] mats)
        {
            Initialise();
            var tensor = BuildTensor(mats);
            var named = NamedOnnxValue.CreateFromTensor(inputName, tensor);

            inputs.Clear(); inputs.Add(named);
            using (var results = session.Run(inputs))
            {
                var outTensor = results.First().AsTensor<float>();
                var flat = outTensor.ToArray();
                // pack into a 1×1 Mat with BatchSize channels
                var outMat = new Mat(1, 1, Depth.F32, BatchSize);
                unsafe
                {
                    float* dst = (float*)outMat.Data.ToPointer();
                    for (int ch = 0; ch < BatchSize; ch++)
                        dst[ch] = flat[ch];
                }
                return outMat;
            }
        }

        ~RippleDetectorRealTime()
        {
            if (bufferPin.IsAllocated) bufferPin.Free();
            session?.Dispose();
        }
    }
}
