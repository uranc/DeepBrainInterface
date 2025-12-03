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
    [Description("Zero-Allocation ONNX inference. Hardcoded for single-threaded low-latency CPU execution.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorRealTime
    {
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; }
            = @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";

        // REMOVED: public int NumThreads { get; set; }

        public int BatchSize { get; set; } = 1;
        public int TimePoints { get; set; } = 92;
        public int Channels { get; set; } = 8;

        // ONNX Resources
        InferenceSession session;
        float[] inputBuffer;
        GCHandle bufferPin;

        // CACHED OBJECTS (Zero Allocation)
        List<NamedOnnxValue> inputContainer;
        DenseTensor<float> inputTensor;

        void Initialise()
        {
            if (session != null) return;

            // 1. Allocate Data Buffer (Pinned)
            inputBuffer = new float[BatchSize * TimePoints * Channels];
            bufferPin = GCHandle.Alloc(inputBuffer, GCHandleType.Pinned);

            // 2. Session Options (Latency Tuned)
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,

                // HARDCODED OPTIMIZATION: 
                // Single-thread is fastest for small matrices (<100 timepoints).
                // Eliminates thread-pool synchronization overhead.
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,

                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableCpuMemArena = true
            };
            options.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

            session = new InferenceSession(ModelPath, options);
            var inputName = session.InputMetadata.Keys.First();

            // 3. Create Tensor Wrapper ONCE
            // 'inputTensor' wraps 'inputBuffer'. Updates to the array are reflected automatically.
            inputTensor = new DenseTensor<float>(inputBuffer, new[] { BatchSize, TimePoints, Channels });

            // 4. Create Input Container ONCE
            inputContainer = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            // Warm-up
            using (var _ = session.Run(inputContainer)) { }
        }
        private Mat RunInference(params Mat[] mats)
        {
            Initialise();

            // STEP 1: Fast Memory Copy (Mat -> Pinned Buffer)
            unsafe
            {
                float* dstBase = (float*)bufferPin.AddrOfPinnedObject().ToPointer();
                int stride = TimePoints * Channels;

                for (int i = 0; i < mats.Length; i++)
                {
                    float* src = (float*)mats[i].Data.ToPointer();
                    float* dst = dstBase + (i * stride);

                    if (mats[i].Rows == TimePoints && mats[i].Cols == Channels)
                    {
                        Buffer.MemoryCopy(src, dst, stride * sizeof(float), stride * sizeof(float));
                    }
                    else if (mats[i].Rows == Channels && mats[i].Cols == TimePoints)
                    {
                        for (int c = 0; c < Channels; c++)
                        {
                            int c_offset = c * TimePoints;
                            for (int t = 0; t < TimePoints; t++)
                            {
                                dst[t * Channels + c] = src[c_offset + t];
                            }
                        }
                    }
                    else throw new ArgumentException($"Unexpected shape");
                }
            }

            // STEP 2: Run Inference
            using (var results = session.Run(inputContainer))
            {
                // STEP 3: Zero-Copy Output
                // FIX: Cast to DenseTensor to access .Buffer
                var outTensor = results.First().AsTensor<float>() as DenseTensor<float>;

                var outMat = new Mat(1, 1, Depth.F32, BatchSize);

                unsafe
                {
                    float* dst = (float*)outMat.Data.ToPointer();

                    // Access via Span (Fastest, avoids .ToArray())
                    var outSpan = outTensor.Buffer.Span;

                    for (int ch = 0; ch < BatchSize; ch++)
                    {
                        dst[ch] = outSpan[ch];
                    }
                }
                return outMat;
            }
        }
        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(m => RunInference(m));
        }

        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return source.Select(t => RunInference(t.Item1, t.Item2));
        }

        ~RippleDetectorRealTime()
        {
            if (bufferPin.IsAllocated) bufferPin.Free();
            session?.Dispose();
        }
    }
}