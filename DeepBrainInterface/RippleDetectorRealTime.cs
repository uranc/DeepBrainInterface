using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Windows.Forms.Design;

namespace DeepBrainInterface
{
    //public enum OnnxProvider { Cpu, Cuda, TensorRT }

    [Combinator]
    [Description("Ultra-low-latency ONNX inference (1×92×8 → 1 float) with per-call timing.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorRealTime : Transform<Mat, Mat>
    {
        /* ───── user parameters ──────────────────────────────────────── */

        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } =
            @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";

        public OnnxProvider Provider { get; set; } = OnnxProvider.Cpu;

        [Description("Set >1 only if you deliberately want micro-batching.")]
        public int BatchSize { get; set; } = 1;

        /* fixed model IO shape ------------------------------------------------ */
        const int T = 92;   // time-points
        const int C = 8;    // channels
        const int OUT_LEN = 1;   // scalar output
        /* --------------------------------------------------------------------- */

        /* ───── internal state ─────────────────────────────────────────────── */

        InferenceSession session;
        string inName, outName;

        // pinned reusable input buffer
        float[] buf = new float[T * C];   // BatchSize is 1 → keep it small
        GCHandle bufPin;

        // one reusable container per Run()
        readonly List<NamedOnnxValue> container = new List<NamedOnnxValue>(1);

        /* timing */
        static readonly Stopwatch swRun = new Stopwatch();
        static readonly Stopwatch swLoop = Stopwatch.StartNew();
        static long prevTicks;
        const int kPrint = 1000;
        int nAccum; double accRun, accCall;

        /* ───── one-time init ─────────────────────────────────────────────── */

        void Initialise()
        {
            if (session != null) return;

            Console.WriteLine("Compiled providers  : " +
                string.Join(", ", OrtEnv.Instance().GetAvailableProviders()));
            if (!bufPin.IsAllocated) bufPin = GCHandle.Alloc(buf, GCHandleType.Pinned);

            var so = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = 4,   // tune for your CPU; 2-6 is usually best
                InterOpNumThreads = 1
            };
            so.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

            try
            {
                if (Provider == OnnxProvider.Cuda)
                    so.AppendExecutionProvider_CUDA(0);
                else if (Provider == OnnxProvider.TensorRT)
                {
                    so.AppendExecutionProvider_CUDA(0);
                    so.AppendExecutionProvider_Tensorrt();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("GPU/TRT attach failed → CPU fallback: " + ex.Message);
            }

            session = new InferenceSession(ModelPath, so);

            inName = session.InputMetadata.First().Key;
            outName = session.OutputMetadata.First().Key;

            Console.WriteLine("ORT session ready. Providers = {0}",
                string.Join(", ", OrtEnv.Instance().GetAvailableProviders()));

            /* one warm-up run (avoids first-call JIT hit) */
            var warm = new DenseTensor<float>(buf, new[] { 1, T, C });
            using (var dummy = session.Run(
                    new[] { NamedOnnxValue.CreateFromTensor(inName, warm) })) { }
        }

        /* ───── fast path: Mat → pinned array → tensor ───────────────────── */

        DenseTensor<float> BuildInputTensor(Mat m)
        {
            unsafe
            {
                float* src = (float*)m.Data.ToPointer();
                float* dst = (float*)bufPin.AddrOfPinnedObject().ToPointer();

                if (m.Rows == C && m.Cols == T)
                {
                    /* transpose to [T, C] in place */
                    for (int c = 0; c < C; ++c)
                        for (int t = 0; t < T; ++t)
                            dst[t * C + c] = src[c * T + t];
                }
                else if (m.Rows == T && m.Cols == C)
                {
                    /* already correct layout – raw memcpy */
                    Buffer.MemoryCopy(src, dst, buf.Length * sizeof(float),
                                      buf.Length * sizeof(float));
                }
                else throw new ArgumentException(
                         $"Unexpected Mat shape {m.Rows}×{m.Cols}; expected 8×92 or 92×8");
            }

            return new DenseTensor<float>(buf, new[] { 1, T, C });
        }

        /* ───── Bonsai pipeline entry point ──────────────────────────────── */

        public override IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Buffer(BatchSize).Select(batch =>
            {
                Initialise();                         // lazy, one-time

                DenseTensor<float> input = BuildInputTensor(batch[0]);

                container.Clear();
                container.Add(NamedOnnxValue.CreateFromTensor(inName, input));

                /* timing – pure ORT execution */
                swRun.Restart();
                var outMat = new Mat(1, 1, Depth.F32, 1);
                using (var res = session.Run(container))
                {
                    swRun.Stop();

                    float value = res.First().AsEnumerable<float>().First();

                    /* prepare 1×1 Mat */
                    outMat.SetReal(0, 0, value);
                }

                /* call-to-call Δt */
                long now = swLoop.ElapsedTicks;
                double usCall = (now - prevTicks) * 1e6 / Stopwatch.Frequency;
                prevTicks = now;

                double usRun = swRun.ElapsedTicks * 1e6 / Stopwatch.Frequency;

                accRun += usRun;
                accCall += usCall;
                if (++nAccum == kPrint)
                {
                    Console.WriteLine($"⟨Δt-run⟩={accRun / kPrint:F1} µs   ⟨Δt-call⟩={accCall / kPrint:F1} µs");
                    nAccum = 0; accRun = accCall = 0;
                }

                return outMat;   // returning dummy; adapt to your downstream needs
            });
        }

        /* ───── cleanup (optional) ──────────────────────────────────────── */

        ~RippleDetectorRealTime()
        {
            if (bufPin.IsAllocated) bufPin.Free();
            session?.Dispose();
        }
    }
}