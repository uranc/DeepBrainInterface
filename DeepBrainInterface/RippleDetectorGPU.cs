using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
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
    [Description("High-Performance GPU Inference (Strict Batch 1 or 2, Pinned Memory).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorGPU
    {
        // ==============================================================================
        // 1. MODEL CONFIGURATION
        // ==============================================================================
        [Category("Model")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("Model")]
        [Description("Strictly 1 (Signal) or 2 (Signal + Artifact).")]
        public int BatchSize { get; set; } = 2;

        [Category("Data Dimensions")]
        public int TimePoints { get; set; } = 44;

        [Category("Data Dimensions")]
        public int Channels { get; set; } = 8;

        // ==============================================================================
        // 2. GPU & THREADING
        // ==============================================================================
        [Category("Execution Provider")]
        public OnnxProvider Provider { get; set; } = OnnxProvider.Cuda;

        [Category("Execution Provider")]
        public int DeviceId { get; set; } = 0;

        [Category("Threading")]
        [Description("Threads for single-op calculation (MatMul).")]
        public int IntraOpNumThreads { get; set; } = 1;

        [Category("Threading")]
        [Description("Threads for parallel sub-graphs.")]
        public int InterOpNumThreads { get; set; } = 1;

        // ==============================================================================
        // 3. INTERNAL RESOURCES
        // ==============================================================================
        private InferenceSession _session;
        private OrtIoBinding _ioBinding;
        private RunOptions _runOptions;

        // Pinned Memory
        private GCHandle _inputPin;
        private GCHandle _outputPin;
        private OrtValue _inputOrtValue;
        private OrtValue _outputOrtValue;

        // Buffers
        private float[] _outputBuffer;
        private int _batchStrideFloats;

        private void Initialise()
        {
            if (_session != null) return;
            if (!File.Exists(ModelPath)) throw new FileNotFoundException("Model not found", ModelPath);

            // A. SESSION CONFIG (GPU Logic)
            var opts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                IntraOpNumThreads = IntraOpNumThreads,
                InterOpNumThreads = InterOpNumThreads
            };

            // Set Environment Vars for TensorRT (If selected)
            if (Provider == OnnxProvider.TensorRT)
            {
                string cacheDir = Path.GetDirectoryName(Path.GetFullPath(ModelPath));
                Environment.SetEnvironmentVariable("ORT_TENSORRT_FP16_ENABLE", "1");
                Environment.SetEnvironmentVariable("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1");
                Environment.SetEnvironmentVariable("ORT_TENSORRT_ENGINE_CACHE_PATH", cacheDir);
            }

            // Append Providers
            try
            {
                if (Provider == OnnxProvider.TensorRT)
                {
                    opts.AppendExecutionProvider_Tensorrt(DeviceId);
                    opts.AppendExecutionProvider_CUDA(DeviceId); // Fallback
                }
                else if (Provider == OnnxProvider.Cuda)
                {
                    opts.AppendExecutionProvider_CUDA(DeviceId);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[GPU Init Failed] {ex.Message}. Falling back to CPU.");
            }

            // Create Session
            _session = new InferenceSession(ModelPath, opts);
            _ioBinding = _session.CreateIoBinding();
            _runOptions = new RunOptions();

            // B. SIZE CALCULATION
            _batchStrideFloats = TimePoints * Channels;
            int totalInputFloats = BatchSize * _batchStrideFloats;

            // C. ALLOCATE PINNED MEMORY
            var inputData = new float[totalInputFloats];
            _inputPin = GCHandle.Alloc(inputData, GCHandleType.Pinned);

            // FIX: Uses BatchSize directly (removed undefined 'outLen')
            _outputBuffer = new float[BatchSize];
            _outputPin = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);

            // D. BIND TENSORS (Zero-Copy)
            var memInfo = OrtMemoryInfo.DefaultInstance; // CPU Pinned Memory -> DMA to GPU
            unsafe
            {
                _inputOrtValue = OrtValue.CreateTensorValueWithData(
                    memInfo, TensorElementType.Float,
                    new long[] { BatchSize, TimePoints, Channels },
                    _inputPin.AddrOfPinnedObject(),
                    totalInputFloats * sizeof(float)
                );

                _outputOrtValue = OrtValue.CreateTensorValueWithData(
                    memInfo, TensorElementType.Float,
                    new long[] { BatchSize, 1 },
                    _outputPin.AddrOfPinnedObject(),
                    _outputBuffer.Length * sizeof(float)
                );
            }

            _ioBinding.BindInput(_session.InputMetadata.Keys.First(), _inputOrtValue);
            _ioBinding.BindOutput(_session.OutputMetadata.Keys.First(), _outputOrtValue);

            // Warmup
            _session.RunWithBinding(_runOptions, _ioBinding);
        }

        // ==============================================================================
        // 4. PROCESSING (Optimized & Robust)
        // ==============================================================================

        // Scenario: Batch 2 (Signal + Artifact)
        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return source.Select(input =>
            {
                Initialise();
                if (BatchSize != 2) throw new InvalidOperationException("GPU Config is Batch 2, but received Tuple.");

                unsafe
                {
                    float* ptr = (float*)_inputPin.AddrOfPinnedObject();

                    // 1. Transpose Signal -> Index 0
                    RobustTransposeCopy(input.Item1, ptr);

                    // 2. Transpose Artifact -> Index [Stride]
                    RobustTransposeCopy(input.Item2, ptr + _batchStrideFloats);
                }

                return RunInference();
            });
        }

        // Scenario: Batch 1 (Signal Only)
        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                Initialise();
                if (BatchSize != 1) throw new InvalidOperationException("GPU Config is Batch 1, but received Single Mat.");

                unsafe
                {
                    RobustTransposeCopy(input, (float*)_inputPin.AddrOfPinnedObject());
                }

                return RunInference();
            });
        }

        // ==============================================================================
        // 5. CORE LOGIC
        // ==============================================================================

        // Hardcoded Transpose: 8x44 (Input) -> 44x8 (Model)
        private unsafe void RobustTransposeCopy(Mat src, float* dstBase)
        {
            if (src.Rows != Channels || src.Cols != TimePoints)
                throw new InvalidOperationException($"Input must be {Channels}x{TimePoints}. Got {src.Rows}x{src.Cols}");

            byte* srcRowPtr = (byte*)src.Data.ToPointer();
            int srcStep = src.Step; // Handles Padding/ROI

            for (int c = 0; c < Channels; c++)
            {
                float* srcFloatRow = (float*)srcRowPtr;
                for (int t = 0; t < TimePoints; t++)
                {
                    // Map [Channel, Time] -> [Time, Channel]
                    dstBase[t * Channels + c] = srcFloatRow[t];
                }
                srcRowPtr += srcStep;
            }
        }

        private Mat RunInference()
        {
            // 1. Execute on GPU (Data auto-transfers via Binding)
            _session.RunWithBinding(_runOptions, _ioBinding);

            // 2. Return Result
            var outMat = new Mat(BatchSize, 1, Depth.F32, 1);
            unsafe
            {
                Marshal.Copy(_outputBuffer, 0, outMat.Data, BatchSize);
            }
            return outMat;
        }

        public void Unload()
        {
            _inputOrtValue?.Dispose();
            _outputOrtValue?.Dispose();
            if (_inputPin.IsAllocated) _inputPin.Free();
            if (_outputPin.IsAllocated) _outputPin.Free();
            _ioBinding?.Dispose();
            _runOptions?.Dispose();
            _session?.Dispose();
            _session = null;
        }
    }
}