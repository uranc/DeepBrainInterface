//using Bonsai;
//using Microsoft.ML.OnnxRuntime;
//using Microsoft.ML.OnnxRuntime.Tensors;
//using OpenCV.Net;
//using System;
//using System.Collections.Generic; // Add this namespace for IReadOnlyList
//using System.ComponentModel;
//using System.Diagnostics;
//using System.Drawing.Design;
//using System.IO; // Add this namespace for FileNotFoundException
//using System.Linq;
//using System.Reactive.Linq;
//using System.Runtime.CompilerServices; // Add this namespace
//using System.Runtime.InteropServices;   // for Marshal.Copy fallback
//using System.Windows.Forms.Design; // Add this namespace
//using System.Reflection; // Add this namespace for accessing runtime properties

//namespace DeepBrainInterface
//{

//    [Combinator]
//    [Description("ONNX ripple detector with Δt-run / Δt-call timing (C# 7.3).")]
//    [WorkflowElementCategory(ElementCategory.Transform)]
//    public class RippleDetectorONNXTimedBackup
//    {
//        /* ───── user parameters ─────────────────────────────────────────── */
//        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
//        public string ModelPath { get; set; } =
//            @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";

//        public OnnxProvider Provider { get; set; } = OnnxProvider.Cpu;
//        public int BatchSize { get; set; } = 1;
//        public int ExpectedTimepoints { get; set; } = 92;
//        public int ExpectedChannels { get; set; } = 8;
//        /* ───────────────────────────────────────────────────────────────── */

//        InferenceSession _session;
//        string _inputName, _outputName;
//        float[] _tensorBuf;          // reusable 8×92 buffer
//        NamedOnnxValue[] _container;

//        /* ───── timing state ────────────────────────────────────────────── */
//        static readonly Stopwatch swRun = new Stopwatch();
//        static readonly Stopwatch swLoop = Stopwatch.StartNew();
//        static long prevTicks;
//        const int kPrint = 1000;
//        int nAccum;
//        double accRun, accCall;
//        /* ───────────────────────────────────────────────────────────────── */
//        void Initialise()
//        {
//            if (_session != null) return;                   // already done

//            /* ---------- build SessionOptions ---------- */
//            var opts = new SessionOptions();

//            opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
//            opts.IntraOpNumThreads = 6;     // try 4–8 and time again
//            opts.InterOpNumThreads = 1;
//            opts.ExecutionMode = ExecutionMode.ORT_PARALLEL;
//            opts.AddSessionConfigEntry("session.use_dnnl", "1");
//            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

//            try
//            {
//                if (Provider == OnnxProvider.Cuda)
//                    opts.AppendExecutionProvider_CUDA(0);
//                else if (Provider == OnnxProvider.TensorRT)
//                {
//                    opts.AppendExecutionProvider_CUDA(0);
//                    opts.AppendExecutionProvider_Tensorrt();
//                }
//            }
//            catch (Exception ex)
//            {
//                Console.WriteLine("GPU / TensorRT load failed → CPU fallback: " + ex.Message);
//                Provider = OnnxProvider.Cpu;
//            }

//            /* ---------- create session ---------- */
//            try
//            {
//                if (!System.IO.File.Exists(ModelPath))
//                    throw new FileNotFoundException("ONNX model not found", ModelPath);

//                _session = new InferenceSession(ModelPath, opts);
//            }
//            catch (Exception ex)
//            {
//                throw new InvalidOperationException("Could not create ORT session", ex);
//            }

//            if (_session == null)
//                throw new InvalidOperationException("ORT returned null session – check DLL versions / PATH.");

//            /* ---------- log active providers ---------- */
//            Console.WriteLine("Providers in session: " +
//                string.Join(", ", _session.GetType()
//                    .GetProperty("ExecutionProviderNames", BindingFlags.Public | BindingFlags.Instance)?
//                    .GetValue(_session) as IReadOnlyList<string> ?? new List<string>()));

//            /* ---------- cache IO names ---------- */
//            foreach (var kv in _session.InputMetadata) { _inputName = kv.Key; break; }
//            foreach (var kv in _session.OutputMetadata) { _outputName = kv.Key; break; }

//            _tensorBuf = new float[ExpectedChannels * ExpectedTimepoints];
//            _container = new NamedOnnxValue[1];
//        }

//        DenseTensor<float> ToTensor(Mat m)
//        {
//            var tensor = new DenseTensor<float>(
//                _tensorBuf,
//                new[] { 1, ExpectedTimepoints, ExpectedChannels });

//            unsafe
//            {
//                float* src = (float*)m.Data.ToPointer();
//                int idx = 0;
//                for (int ch = 0; ch < ExpectedChannels; ++ch)
//                    for (int t = 0; t < ExpectedTimepoints; ++t)
//                        _tensorBuf[idx++] = src[ch * ExpectedTimepoints + t];
//            }
//            return tensor;
//        }

//        public IObservable<Mat> Process(IObservable<Mat> source)
//        {
//            return source.Buffer(BatchSize).Select(batch =>
//            {
//                Initialise();

//                DenseTensor<float> input = ToTensor(batch[0]);
//                _container[0] = NamedOnnxValue.CreateFromTensor(_inputName, input);

//                /* ─── timing (pure inference) ─── */
//                swRun.Restart();
//                IDisposable resultDisp = _session.Run(_container);
//                swRun.Stop();

//                // collect the Tensor before disposing results
//                var results = (IDisposableReadOnlyCollection<DisposableNamedOnnxValue>)resultDisp;
//                Tensor<float> outTensor = results.First().AsTensor<float>();
//                resultDisp.Dispose();

//                long nowTicks = swLoop.ElapsedTicks;
//                long dtTicks = nowTicks - prevTicks;
//                prevTicks = nowTicks;

//                double usRun = swRun.ElapsedTicks * 1e6 / Stopwatch.Frequency;
//                double usCall = dtTicks * 1e6 / Stopwatch.Frequency;

//                accRun += usRun;
//                accCall += usCall;
//                if (++nAccum == kPrint)
//                {
//                    Console.WriteLine(
//                        $"ONNX ⟨Δt-run⟩={accRun / kPrint:F2} µs   ⟨Δt-call⟩={accCall / kPrint:F2} µs");
//                    nAccum = 0; accRun = accCall = 0;
//                }
//                /* ─────────────────────────────── */

//                /* ─── copy tensor → Mat (compatible with ORT ≤1.15) ─── */
//                int len = (int)outTensor.Length;
//                var outMat = new Mat(1, len, Depth.F32, 1);
//                float[] tmp = outTensor.ToArray();   // safest API in 1.12–1.15

//                unsafe
//                {
//                    System.Buffer.MemoryCopy(
//                        Unsafe.AsPointer(ref tmp[0]),
//                        outMat.Data.ToPointer(),
//                        (long)len * sizeof(float),
//                        (long)len * sizeof(float));
//                }
//                return outMat;
//            });
//        }
//    }
//}
