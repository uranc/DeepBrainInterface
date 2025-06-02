using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms.Design; // Add this namespace
using SD = System.Diagnostics;       // avoid collision with our Process(..)

// ───────── pin-thread helpers ─────────
static class Affinity
{
    [DllImport("kernel32.dll")] public static extern IntPtr GetCurrentThread();
    [DllImport("kernel32.dll")]
    public static extern UIntPtr SetThreadAffinityMask(
                                                  IntPtr hThread, UIntPtr mask);
}

namespace DeepBrainInterface
{
    [Combinator]
    [Description("ONNX inference pinned to a dedicated core (High-priority process).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public unsafe class RippleDetectorONNXThreaded
    {
        // ─── public parameters ────────────────────────────────────────────────
        [Editor(typeof(FileNameEditor), typeof(System.Drawing.Design.UITypeEditor))]
        public string ModelPath { get; set; } =
            @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";

        public OnnxProvider Provider { get; set; } = OnnxProvider.Cpu;

        [Description("Windows per batch; use 1 for real-time.")]
        public int BatchSize { get; set; } = 1;

        public string InputName { get; set; } = "";
        public string OutputName { get; set; } = "";

        public int ExpectedTimepoints { get; set; } = 92;
        public int ExpectedChannels { get; set; } = 8;
        // ───────────────────────────────────────────────────────────────────────

        // ONNX & buffers
        InferenceSession _session;
        string _in, _out;

        float[] _workBuf;                         // reused
        readonly List<NamedOnnxValue> _container = new List<NamedOnnxValue>(1);

        Mat _outMat;                              // allocated after 1st run
        bool _ready;                              // one-time guard

        // ─── one-time initialisation ──────────────────────────────────────────
        void InitOnce()
        {
            if (_ready) return; _ready = true;

            // 1) raise entire process priority
            SD.Process.GetCurrentProcess().PriorityClass = SD.ProcessPriorityClass.High;

            // 2) pin this thread to P-core #2
            if (Affinity.SetThreadAffinityMask(
                    Affinity.GetCurrentThread(), new UIntPtr(1u << 2)) == UIntPtr.Zero)
                throw new InvalidOperationException("Cannot pin to CPU 2.");

            // 3) create ONNX session
            var opts = new SessionOptions();
            if (Provider == OnnxProvider.Cuda) opts.AppendExecutionProvider_CUDA(0);
            if (Provider == OnnxProvider.TensorRT)
            {
                opts.AppendExecutionProvider_CUDA(0);
                opts.AppendExecutionProvider_Tensorrt();
            }

            _session = new InferenceSession(ModelPath, opts);

            _in = string.IsNullOrEmpty(InputName) ? _session.InputMetadata.First().Key : InputName;
            _out = string.IsNullOrEmpty(OutputName) ? _session.OutputMetadata.First().Key : OutputName;
        }

        // ─── IList<Mat>  → DenseTensor<float>  [B×T×C] ───────────────────────
        DenseTensor<float> ToTensor(IList<Mat> mats)
        {
            int B = mats.Count;
            int total = B * ExpectedTimepoints * ExpectedChannels;
            if (_workBuf == null || _workBuf.Length < total) _workBuf = new float[total];

            for (int b = 0; b < B; ++b)
            {
                Mat m = mats[b];
                float* src = (float*)m.Data.ToPointer();
                fixed (float* dst0 = &_workBuf[b * ExpectedTimepoints * ExpectedChannels])
                {
                    float* dst = dst0;

                    // Fast path: already [C x T] and we need transpose
                    for (int c = 0; c < ExpectedChannels; ++c)
                    {
                        float* rowSrc = src + c * ExpectedTimepoints;
                        for (int t = 0; t < ExpectedTimepoints; ++t)
                            dst[t * ExpectedChannels + c] = rowSrc[t];
                    }
                }
            }
            return new DenseTensor<float>(_workBuf,
                   new[] { B, ExpectedTimepoints, ExpectedChannels });
        }

        // ─── Bonsai combinator entry ──────────────────────────────────────────
        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source
                .Buffer(BatchSize)                       // BatchSize=1 for RT
                .Select(batch =>
                {
                    InitOnce();

                    var tensor = ToTensor(batch);
                    _container.Clear();
                    _container.Add(NamedOnnxValue.CreateFromTensor(_in, tensor));

                    using (var results = _session.Run(_container))
                    {
                        var t = results.First().AsTensor<float>();

#if ONNX_PIN_AVAILABLE     // <-- define this if your onnxruntime ≥1.16
                        var handle = t.Buffer.Pin();
                        try
                        {
                            if (_outMat == null || _outMat.Cols != t.Length)
                                _outMat = new Mat(1, t.Length, Depth.F32, 1);

                            Buffer.MemoryCopy((void*)handle.Pointer,
                                              _outMat.Data.ToPointer(),
                                              t.Length * sizeof(float),
                                              t.Length * sizeof(float));
                        }
                        finally { handle.Dispose(); }
#else
                        float[] arr = t.ToArray();          // unavoidable copy
                        if (_outMat == null || _outMat.Cols != arr.Length)
                            _outMat = new Mat(1, arr.Length, Depth.F32, 1);

                        fixed (float* src = arr)
                        {
                            Buffer.MemoryCopy(src,
                                              _outMat.Data.ToPointer(),
                                              arr.Length * sizeof(float),
                                              arr.Length * sizeof(float));
                        }
#endif
                    }
                    return _outMat;
                });
        }
    }
}
