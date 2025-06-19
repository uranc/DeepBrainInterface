//using Bonsai;
//using Bonsai.Reactive;
//using Microsoft.ML.OnnxRuntime;
//using Microsoft.ML.OnnxRuntime.Tensors;
//using OpenCV.Net;
//using System;
//using System.Collections.Generic;
//using System.ComponentModel;
//using System.Drawing.Design;
//using System.Linq;
//using System.Reactive.Linq;
//using System.Runtime.CompilerServices;
//using System.Runtime.InteropServices;
//using System.Windows.Forms.Design;

//namespace DeepBrainInterface
//{
//    public enum OnnxProvider { Cpu, Cuda, TensorRT }

//    [Combinator]
//    [Description("Runs an ONNX ripple detector on streaming Mat data (CPU / CUDA / TensorRT).")]
//    [WorkflowElementCategory(ElementCategory.Transform)]
//    public class RippleDetectorONNXProbe : Transform<Mat, Mat>
//    {
//        /* ───────────── user parameters ─────────────────────────────── */
//        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
//        public string ModelPath { get; set; } =
//            @"C:\Models\ripple_detector.onnx";

//        public OnnxProvider Provider { get; set; } = OnnxProvider.Cpu;
//        public int BatchSize { get; set; } = 1;
//        public int ExpectedTimepoints { get; set; } = 92;
//        public int ExpectedChannels { get; set; } = 8;
//        /* ───────────────────────────────────────────────────────────── */

//        InferenceSession _session;
//        string _inputName, _outputName;

//        /* scratch buffers re-used every call */
//        float[] _flatBuffer;
//        readonly List<NamedOnnxValue> _container = new List<NamedOnnxValue>(1);

//        /* one-time setup (lazy) */
//        void Initialise()
//        {
//            if (_session != null) return;

//            Console.WriteLine("Compiled providers  : "
//                + string.Join(", ", OrtEnv.Instance().GetAvailableProviders()));
//            Console.WriteLine("Requested provider  : " + Provider);

//            var so = new SessionOptions();
//            try
//            {
//                if (Provider == OnnxProvider.Cuda)
//                    so.AppendExecutionProvider_CUDA(0);
//                else if (Provider == OnnxProvider.TensorRT)
//                {
//                    so.AppendExecutionProvider_CUDA(0);
//                    so.AppendExecutionProvider_Tensorrt();
//                }
//            }
//            catch (Exception ex)
//            {
//                Console.WriteLine("GPU / TRT attach failed → CPU fallback: " + ex.Message);
//            }

//            _session = new InferenceSession(ModelPath, so);

//            _inputName = _session.InputMetadata.First().Key;
//            _outputName = _session.OutputMetadata.First().Key;

//            _flatBuffer = new float[BatchSize * ExpectedChannels * ExpectedTimepoints];

//            Console.WriteLine("?  Session initialised (check ORT log above for EP binding).");
//        }

//        /* -------- helper: copy OpenCV Mats → DenseTensor<float> ------- */
//        DenseTensor<float> ToTensor(List<Mat> mats)
//        {
//            int batch = mats.Count;
//            int elems = batch * ExpectedChannels * ExpectedTimepoints;
//            if (_flatBuffer.Length < elems) _flatBuffer = new float[elems];

//            unsafe
//            {
//                for (int b = 0; b < batch; ++b)
//                {
//                    var m = mats[b];
//                    float* s = (float*)m.Data.ToPointer();
//                    int dst0 = b * ExpectedChannels * ExpectedTimepoints;

//                    for (int ch = 0; ch < ExpectedChannels; ++ch)
//                    {
//                        for (int t = 0; t < ExpectedTimepoints; ++t)
//                        {
//                            // transpose:  [ch, t]  → flat[ t * C + ch ]
//                            _flatBuffer[dst0 + t * ExpectedChannels + ch] =
//                                s[ch * ExpectedTimepoints + t];
//                        }
//                    }
//                }
//            }

//            var dims = new[] { batch, ExpectedTimepoints, ExpectedChannels };
//            var tensor = new DenseTensor<float>(dims);
//            for (int i = 0; i < tensor.Length; ++i) tensor[i] = _flatBuffer[i];
//            return tensor;
//        }

//        public override IObservable<Mat> Process(IObservable<Mat> source)
//        {
//            return source.Buffer(BatchSize).Select(mats =>
//            {
//                Initialise();

//                var input = ToTensor(mats.ToList());
//                _container.Clear();
//                _container.Add(NamedOnnxValue.CreateFromTensor(_inputName, input));

//                using (var results = _session.Run(_container))
//                {
//                    var outTensor = results.First().AsTensor<float>();
//                    int len = (int)outTensor.Length;

//                    var outMat = new Mat(1, len, Depth.F32, 1);

//                    /* copy tensor → Mat using SetReal */
//                    float[] tmp = outTensor.ToArray();
//                    for (int i = 0; i < len; i++)
//                    {
//                        outMat.SetReal(i, tmp[i]);
//                    }
//                    return outMat;
//                }
//            });
//        }
//    }
//}
