using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Drawing.Design;
using System.Windows.Forms.Design;

namespace DeepBrainInterface
{
    public enum OnnxProvider { Cpu, Cuda, TensorRT }

    [Combinator]
    [Description("High-Performance GPU Inference with Tunable Threading & TensorRT Caching.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorGPU
    {
        /* ───── User Parameters ─────────────────────────────────────────── */

        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Description("Execution Provider. TensorRT requires engine build (slow startup on first run).")]
        public OnnxProvider Provider { get; set; } = OnnxProvider.Cuda;

        [Description("GPU Device ID (usually 0).")]
        public int DeviceId { get; set; } = 0;

        [Description("Number of threads to parallelize the execution within nodes. Increase for large models.")]
        public int IntraOpNumThreads { get; set; } = 1;

        [Description("Number of threads to parallelize the execution of the graph (across nodes).")]
        public int InterOpNumThreads { get; set; } = 1;

        [Description("Input TimePoints (must match model training, e.g., 92).")]
        public int TimePoints { get; set; } = 92;

        public int Channels { get; set; } = 8;
        /* ───────────────────────────────────────────────────────────────── */

        // ---- ONNX RESOURCES ----
        InferenceSession _session;
        OrtIoBinding _binding;
        RunOptions _runOptions;
        OrtMemoryInfo _memInfo;

        string _inputName;
        string _outputName;

        // ---- PINNED MEMORY BUFFERS ----
        float[] _inputBuffer;
        GCHandle _inputPin;
        OrtValue _inputOrtValue;

        float[] _outputBuffer;
        GCHandle _outputPin;
        OrtValue _outputOrtValue;

        // State tracking
        int _currentCapacity = 0;
        int _activeBatchSize = 0;

        private void InitializeSession()
        {
            if (_session != null) return;
            if (!File.Exists(ModelPath)) throw new FileNotFoundException("Model not found", ModelPath);

            // 1. Configure General Options
            var opts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                IntraOpNumThreads = IntraOpNumThreads,
                InterOpNumThreads = InterOpNumThreads
            };

            // 2. Configure Providers with Fallback
            try
            {
                if (Provider == OnnxProvider.TensorRT)
                {
                    // STRATEGY: Use Environment Variables for Advanced Config
                    // This bypasses C# wrapper version incompatibilities.
                    string cacheDir = Path.GetDirectoryName(ModelPath);

                    // Force FP16 precision (Massive speedup on Tensor Cores)
                    Environment.SetEnvironmentVariable("ORT_TENSORRT_FP16_ENABLE", "1");

                    // Enable Engine Caching (Critical: prevents rebuilding engine every run)
                    Environment.SetEnvironmentVariable("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1");
                    Environment.SetEnvironmentVariable("ORT_TENSORRT_ENGINE_CACHE_PATH", cacheDir);

                    // Use the Simple Integer API (Universally supported)
                    opts.AppendExecutionProvider_Tensorrt(DeviceId);

                    // Fallback to CUDA
                    opts.AppendExecutionProvider_CUDA(DeviceId);
                }
                else if (Provider == OnnxProvider.Cuda)
                {
                    opts.AppendExecutionProvider_CUDA(DeviceId);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[GPU Error] {ex.Message}. Falling back to CPU.");
                Provider = OnnxProvider.Cpu;
            }

            // 3. Create Session & IO Binding
            _session = new InferenceSession(ModelPath, opts);
            _memInfo = OrtMemoryInfo.DefaultInstance; // CPU Pinned Memory

            _binding = _session.CreateIoBinding();
            _runOptions = new RunOptions();

            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();
        }

        private void PrepareBinding(int batchSize)
        {
            InitializeSession();

            bool capacityChanged = batchSize > _currentCapacity;
            bool shapeChanged = batchSize != _activeBatchSize;

            // Resize Physical Memory
            if (capacityChanged)
            {
                CleanUpOrtValues();
                if (_inputPin.IsAllocated) _inputPin.Free();
                if (_outputPin.IsAllocated) _outputPin.Free();

                _currentCapacity = Math.Max(batchSize, _currentCapacity);

                int inSize = _currentCapacity * TimePoints * Channels;
                _inputBuffer = new float[inSize];
                _inputPin = GCHandle.Alloc(_inputBuffer, GCHandleType.Pinned);

                // Assuming Output is [Batch, 1]
                int outSize = _currentCapacity;
                _outputBuffer = new float[outSize];
                _outputPin = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);
            }

            // Update Binding if Shape Changed
            if (capacityChanged || shapeChanged)
            {
                CleanUpOrtValues();

                long[] inShape = new long[] { batchSize, TimePoints, Channels };
                long[] outShape = new long[] { batchSize, 1 };

                unsafe
                {
                    // Create Native Wrappers (Zero-Copy)
                    _inputOrtValue = OrtValue.CreateTensorValueWithData(
                        _memInfo,
                        TensorElementType.Float,
                        inShape,
                        _inputPin.AddrOfPinnedObject(),
                        batchSize * TimePoints * Channels * sizeof(float)
                    );

                    _outputOrtValue = OrtValue.CreateTensorValueWithData(
                        _memInfo,
                        TensorElementType.Float,
                        outShape,
                        _outputPin.AddrOfPinnedObject(),
                        batchSize * sizeof(float)
                    );
                }

                _binding.ClearBoundInputs();
                _binding.ClearBoundOutputs();
                _binding.BindInput(_inputName, _inputOrtValue);
                _binding.BindOutput(_outputName, _outputOrtValue);

                _activeBatchSize = batchSize;
            }
        }

        private void CleanUpOrtValues()
        {
            _inputOrtValue?.Dispose();
            _inputOrtValue = null;
            _outputOrtValue?.Dispose();
            _outputOrtValue = null;
        }

        private Mat RunInference(IList<Mat> mats)
        {
            int batchSize = mats.Count;
            if (batchSize == 0) return null;

            // 1. Setup
            PrepareBinding(batchSize);

            // 2. Copy Data (CPU -> Pinned CPU)
            unsafe
            {
                float* dstBase = (float*)_inputPin.AddrOfPinnedObject().ToPointer();
                int stride = TimePoints * Channels;
                long bytesPerMat = stride * sizeof(float);

                for (int i = 0; i < batchSize; i++)
                {
                    float* src = (float*)mats[i].Data.ToPointer();
                    float* dst = dstBase + (i * stride);

                    if (mats[i].Rows == TimePoints && mats[i].Cols == Channels)
                    {
                        Buffer.MemoryCopy(src, dst, bytesPerMat, bytesPerMat);
                    }
                    else if (mats[i].Rows == Channels && mats[i].Cols == TimePoints)
                    {
                        // Transpose
                        for (int c = 0; c < Channels; c++)
                        {
                            int cOff = c * TimePoints;
                            for (int t = 0; t < TimePoints; t++)
                            {
                                dst[t * Channels + c] = src[cOff + t];
                            }
                        }
                    }
                    else
                    {
                        Buffer.MemoryCopy(src, dst, bytesPerMat, bytesPerMat);
                    }
                }
            }

            // 3. Execute (Pinned CPU -> GPU VRAM -> Compute -> GPU VRAM -> Pinned CPU)
            _session.RunWithBinding(_runOptions, _binding);

            // 4. Return Result
            var outMat = new Mat(batchSize, 1, Depth.F32, 1);
            unsafe
            {
                float* resultPtr = (float*)outMat.Data.ToPointer();
                Marshal.Copy(_outputBuffer, 0, (IntPtr)resultPtr, batchSize);
            }

            return outMat;
        }

        // ---- OVERLOADS ----

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(m => RunInference(new[] { m }));
        }

        public IObservable<Mat> Process(IObservable<IList<Mat>> source)
        {
            return source.Select(batch => RunInference(batch));
        }

        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return source.Select(t => RunInference(new[] { t.Item1, t.Item2 }));
        }

        public void Unload()
        {
            CleanUpOrtValues();
            if (_inputPin.IsAllocated) _inputPin.Free();
            if (_outputPin.IsAllocated) _outputPin.Free();
            _binding?.Dispose();
            _runOptions?.Dispose();
            _session?.Dispose();
        }
    }
}