//using Bonsai;
//using Microsoft.ML.OnnxRuntime;
//using Microsoft.ML.OnnxRuntime.Tensors;
//using OpenCV.Net;
//using System;
//using System.Collections.Generic;
//using System.ComponentModel;
//using System.Drawing.Design;
//using System.Linq;
//using System.Reactive.Linq;
//using System.Runtime.InteropServices;
//using System.Windows.Forms.Design;

//namespace DeepBrainInterface
//{
//    [Combinator]
//    [Description("Ultra-low-latency ONNX inference (1×T×C → 1 float)")]
//    [WorkflowElementCategory(ElementCategory.Transform)]
//    public class RippleDetectorRealTimeBackup : Transform<Mat, Mat>
//    {
//        /* ───── user parameters ──────────────────────────────────────── */

//        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
//        public string ModelPath { get; set; } =
//            @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";

//        [Description("Number of threads for ONNX Runtime inference (2-6 typically best)")]
//        public int NumThreads { get; set; } = 5;

//        [Description("Set >1 only if you deliberately want micro-batching.")]
//        public int BatchSize { get; set; } = 1;

//        [Description("Number of time-points in the input window")]
//        public int TimePoints { get; set; } = 92;

//        [Description("Number of channels in the input data")]
//        public int Channels { get; set; } = 8;

//        const int OUT_LEN = 1;   // scalar output

//        /* ───── internal state ─────────────────────────────────────────────── */

//        InferenceSession session;
//        string inName, outName;

//        // pinned reusable input buffer
//        float[] buf;
//        GCHandle bufPin;

//        // one reusable container per Run()
//        readonly List<NamedOnnxValue> container = new List<NamedOnnxValue>(1);

//        // pre-allocated output tensor
//        float[] outValue;

//        /* ───── one-time init ─────────────────────────────────────────────── */

//        void Initialise()
//        {
//            if (session != null) return;

//            buf = new float[TimePoints * Channels];

//            if (!bufPin.IsAllocated) bufPin = GCHandle.Alloc(buf, GCHandleType.Pinned);

//            var so = new SessionOptions
//            {
//                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
//                IntraOpNumThreads = NumThreads,
//                InterOpNumThreads = 1,
//                EnableCpuMemArena = true
//            };

//            so.AddSessionConfigEntry("session.enable_mem_pattern", "1");
//            so.AddSessionConfigEntry("session.execution_mode", "ORT_SEQUENTIAL");
//            so.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
//            so.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

//            session = new InferenceSession(ModelPath, so);

//            inName = session.InputMetadata.First().Key;
//            outName = session.OutputMetadata.First().Key;

//            /* Pre-allocate output tensor to avoid allocation during inference */
//            outValue = new float[OUT_LEN];

//            /* one warm-up run (avoids first-call JIT hit) */
//            var warm = new DenseTensor<float>(buf, new[] { 1, TimePoints, Channels });
//            using (var dummy = session.Run(
//                    new[] { NamedOnnxValue.CreateFromTensor(inName, warm) })) { }
//        }

//        /* ───── fast path: Mat → pinned array → tensor ───────────────────── */

//        DenseTensor<float> BuildInputTensor(Mat m)
//        {
//            unsafe
//            {
//                float* src = (float*)m.Data.ToPointer();
//                float* dst = (float*)bufPin.AddrOfPinnedObject().ToPointer();

//                // Optimize for the most common case (C×T, needing transpose)
//                if (m.Rows == Channels && m.Cols == TimePoints)
//                {
//                    /* transpose to [T, C] in place - unrolled version */
//                    for (int c = 0; c < Channels; ++c)
//                    {
//                        int srcOffset = c * TimePoints;
//                        for (int t = 0; t < TimePoints; ++t)
//                        {
//                            dst[t * Channels + c] = src[srcOffset + t];
//                        }
//                    }
//                }
//                else if (m.Rows == TimePoints && m.Cols == Channels)
//                {
//                    /* already correct layout – raw memcpy */
//                    Buffer.MemoryCopy(src, dst, buf.Length * sizeof(float),
//                                      buf.Length * sizeof(float));
//                }
//                else throw new ArgumentException(
//                         $"Unexpected Mat shape {m.Rows}×{m.Cols}; expected {Channels}×{TimePoints} or {TimePoints}×{Channels}");
//            }

//            return new DenseTensor<float>(buf, new[] { 1, TimePoints, Channels });
//        }

//        /* ───── Bonsai pipeline entry point ──────────────────────────────── */

//        public override IObservable<Mat> Process(IObservable<Mat> source)
//        {
//            return source.Buffer(BatchSize).Select(batch =>
//            {
//                Initialise();  // lazy, one-time

//                DenseTensor<float> input = BuildInputTensor(batch[0]);

//                // Reuse container without clearing/re-adding when possible
//                if (container.Count == 0)
//                {
//                    container.Add(NamedOnnxValue.CreateFromTensor(inName, input));
//                }
//                else
//                {
//                    container[0] = NamedOnnxValue.CreateFromTensor(inName, input);
//                }

//                var outMat = new Mat(1, 1, Depth.F32, 1);
//                using (var res = session.Run(container))
//                {
//                    outValue[0] = res.First().AsTensor<float>().FirstOrDefault();
//                    outMat.SetReal(0, 0, outValue[0]);
//                }

//                return outMat;
//            });
//        }

//        /* ───── cleanup (optional) ──────────────────────────────────────── */

//        ~RippleDetectorRealTimeBackup()
//        {
//            if (bufPin.IsAllocated) bufPin.Free();
//            session?.Dispose();
//        }
//    }
//}