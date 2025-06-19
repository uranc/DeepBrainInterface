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
    public enum OnnxProvider { Cpu, Cuda, TensorRT }

    [Combinator]
    [Description("Runs an ONNX model (CPU / CUDA / TensorRT) on streaming Mat data.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorONNX
    {
        /* ─────── User parameters ───────────────────────────────────── */
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } =
            @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";

        public OnnxProvider Provider { get; set; } = OnnxProvider.Cpu;
        public int BatchSize { get; set; } = 1;
        public string InputName { get; set; } = "";
        public string OutputName { get; set; } = "";
        public int ExpectedTimepoints { get; set; } = 92;
        public int ExpectedChannels { get; set; } = 8;
        /* ────────────────────────────────────────────────────────────── */

        InferenceSession _session;
        string _inputName, _outputName;
        float[] _reuseBuffer;
        readonly List<NamedOnnxValue> _container = new List<NamedOnnxValue>(1);

        /* ───── Initialise ORT session lazily ───────────────────────── */
        void Initialize()
        {
            if (_session != null) return;

            var so = new SessionOptions();

            try
            {
                switch (Provider)
                {
                    case OnnxProvider.Cuda:
                        so.AppendExecutionProvider_CUDA(0);
                        break;

                    case OnnxProvider.TensorRT:
                        so.AppendExecutionProvider_CUDA(0);
                        so.AppendExecutionProvider_Tensorrt();
                        break;

                    case OnnxProvider.Cpu:
                    default:
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠ GPU / TensorRT attach failed → CPU fallback: {ex.Message}");
            }

            _session = new InferenceSession(ModelPath, so);
            _inputName = string.IsNullOrEmpty(InputName)
                               ? _session.InputMetadata.First().Key : InputName;
            _outputName = string.IsNullOrEmpty(OutputName)
                               ? _session.OutputMetadata.First().Key : OutputName;

            Console.WriteLine("Compiled EPs : " +
                string.Join(", ", OrtEnv.Instance().GetAvailableProviders()));
        }

        /* ───── Mat[channels×time] → 3-D tensor (B,T,C) ─────────────── */
        DenseTensor<float> ToTensor(IList<Mat> mats)
        {
            int B = mats.Count;
            int N = ExpectedTimepoints;
            int C = ExpectedChannels;
            int len = B * N * C;

            if (_reuseBuffer == null || _reuseBuffer.Length < len)
                _reuseBuffer = new float[len];

            unsafe
            {
                for (int b = 0; b < B; ++b)
                {
                    float* src = (float*)mats[b].Data.ToPointer();
                    fixed (float* dst0 = &_reuseBuffer[b * N * C])
                    {
                        // reshape + transpose (channels,row) → (time,channel)
                        for (int ch = 0; ch < C; ++ch)
                            for (int t = 0; t < N; ++t)
                                dst0[t * C + ch] = src[ch * N + t];
                    }
                }
            }
            
            // Create tensor using array constructor instead of ReadOnlySpan
            var tensor = new DenseTensor<float>(_reuseBuffer, new[] { B, N, C });
            
            return tensor;
        }

        /* ───── Streaming processing node ───────────────────────────── */
        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Buffer(BatchSize)
                         .Select(batch =>
                         {
                             Initialize();

                             var tensor = ToTensor(batch);
                             _container.Clear();
                             _container.Add(NamedOnnxValue.CreateFromTensor(_inputName, tensor));

                             var results = _session.Run(_container);
                             try
                             {
                                 var prob = results.First().AsEnumerable<float>().First();

                                 // return a 1×1 Mat with the probability
                                 var outMat = new Mat(1, 1, Depth.F32, 1);
                                 Marshal.Copy(new[] { prob }, 0, outMat.Data, 1);
                                 return outMat;
                             }
                             finally
                             {
                                 results.Dispose();
                             }
                         });
        }
    }
}
