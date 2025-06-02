using Bonsai;
using OpenCV.Net;
using System;
using System.Collections.Generic; // Add this namespace
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms.Design;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DeepBrainInterface
{
    //public enum OnnxProvider { Cpu, Cuda, TensorRT }

    public enum OnnxProvider
    {
        Cpu,
        Cuda,
        TensorRT
    }  
[Combinator]
    [Description("Runs an ONNX model on streaming Mat data.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorONNX
    {
        // ---------- user-configurable properties ----------
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        [Description("Path to the ONNX model file (*.onnx)")]
        public string ModelPath { get; set; } =
            @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";

        [Description("Execution provider to use (CPU, CUDA, TensorRT)")]
        public OnnxProvider Provider { get; set; } = OnnxProvider.Cpu;

        [Description("How many incoming frames to batch together. " +
                     "Use 1 for pure frame-by-frame, >1 for micro-batching.")]
        public int BatchSize { get; set; } = 1;

        [Description("Name of the graph input tensor (leave empty to take the first).")]
        public string InputName { get; set; } = "";

        [Description("Name of the graph output tensor (leave empty to take the first).")]
        public string OutputName { get; set; } = "";

        [Description("Number of timepoints the model expects")]
        public int ExpectedTimepoints { get; set; } = 1104;

        [Description("Number of channels the model expects")]
        public int ExpectedChannels { get; set; } = 8;
        // ---------------------------------------------------

        InferenceSession _session;
        string _inputName;
        string _outputName;

        // Add field for buffer reuse
        private float[] _reuseBuffer;
        private List<NamedOnnxValue> _containerCache;

        // One-time initialization – called lazily from Process
        void Initialize()
        {
            if (_session != null) return;

            var opts = new SessionOptions();
            try
            {
                switch (Provider)
                {
                    case OnnxProvider.Cuda:
                        try
                        {
                            opts.AppendExecutionProvider_CUDA(0);
                        }
                        catch (Exception ex)
                        {
                            throw new InvalidOperationException(
                                "CUDA initialization failed. Verify CUDA toolkit and GPU drivers are installed.", ex);
                        }
                        break;
                    case OnnxProvider.TensorRT:
                        try
                        {
                            opts.AppendExecutionProvider_CUDA(0);
                            opts.AppendExecutionProvider_Tensorrt();
                        }
                        catch (Exception ex)
                        {
                            throw new InvalidOperationException(
                                "TensorRT initialization failed. Verify TensorRT and CUDA are properly installed.", ex);
                        }
                        break;
                    case OnnxProvider.Cpu:
                    default:
                        // CPU provider is always available
                        break;
                }

                _session = new InferenceSession(ModelPath, opts);
                _inputName = string.IsNullOrEmpty(InputName)
                                  ? _session.InputMetadata.First().Key
                                  : InputName;
                _outputName = string.IsNullOrEmpty(OutputName)
                                  ? _session.OutputMetadata.First().Key
                                  : OutputName;
            }
            catch (DllNotFoundException ex)
            {
                // If CUDA DLLs are missing, provide a helpful message
                var message = ex.Message.Contains("onnxruntime_providers_cuda")
                    ? "CUDA support DLLs not found. Ensure CUDA toolkit is installed and PATH is set correctly."
                    : ex.Message;
                throw new InvalidOperationException(message, ex);
            }
        }

        // -------- Optimized ToTensor method ----------
        DenseTensor<float> ToTensor(List<Mat> mats)
        {
            var batchSize = mats.Count;
            var totalSize = batchSize * ExpectedTimepoints * ExpectedChannels;

            // Reuse buffer to avoid allocations
            if (_reuseBuffer == null || _reuseBuffer.Length < totalSize)
            {
                _reuseBuffer = new float[totalSize];
            }

            unsafe
            {
                for (int b = 0; b < batchSize; b++)
                {
                    var m = mats[b];

                    // Direct pointer access for faster data copying
                    float* srcPtr = (float*)m.Data.ToPointer();
                    fixed (float* dstPtr = &_reuseBuffer[b * ExpectedTimepoints * ExpectedChannels])
                    {
                        if (m.Rows == ExpectedChannels && m.Cols == ExpectedTimepoints)
                        {
                            // Optimized transpose when dimensions match expectations
                            for (int channel = 0; channel < ExpectedChannels; channel++)
                            {
                                for (int timepoint = 0; timepoint < ExpectedTimepoints; timepoint++)
                                {
                                    dstPtr[timepoint * ExpectedChannels + channel] =
                                        srcPtr[channel * ExpectedTimepoints + timepoint];
                                }
                            }
                        }
                        else
                        {
                            // Direct copy when no transpose needed
                            Buffer.MemoryCopy(srcPtr, dstPtr,
                                totalSize * sizeof(float),
                                ExpectedTimepoints * ExpectedChannels * sizeof(float));
                        }
                    }
                }
            }

            return new DenseTensor<float>(_reuseBuffer, new[] { batchSize, ExpectedTimepoints, ExpectedChannels });
        }

        // -------- Optimized Process method ----------
        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            // Initialize container cache
            _containerCache = new List<NamedOnnxValue>(1);

            return source.Buffer(BatchSize).Select(mats =>
            {
                Initialize();

                var inputTensor = ToTensor(mats.ToList());

                // Reuse container to avoid allocations
                _containerCache.Clear();
                _containerCache.Add(NamedOnnxValue.CreateFromTensor(_inputName, inputTensor));

                using (var results = _session.Run(_containerCache))
                {
                    var outputTensor = results.First().AsTensor<float>();
                    var outputLen = (int)outputTensor.Length;

                    var outputMat = new Mat(1, outputLen, Depth.F32, 1);
                    unsafe
                    {
                        // Direct memory copy for output
                        fixed (float* srcPtr = outputTensor.ToArray())
                        {
                            Buffer.MemoryCopy(srcPtr, outputMat.Data.ToPointer(),
                                outputLen * sizeof(float),
                                outputLen * sizeof(float));
                        }
                    }

                    return outputMat;
                }
            });
        }
    }
}