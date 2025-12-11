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
using System.Runtime.InteropServices;

// --- FIXED: Alias must be at the top level or immediately inside the namespace ---
using PredictionResult = System.ValueTuple<OpenCV.Net.Mat, double>;

namespace DeepBrainInterface
{
    public enum OnnxProvider { Cpu, Cuda, TensorRT }

    [Combinator]
    [Description("High-Performance GPU Inference. Zero-Allocation Mode.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorGPU : IDisposable
    {
        /* ───── User Parameters ─────────────────────────────────────────── */

        [Category("Model")]
        [Description("Path to the ONNX model file.")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        [FileNameFilter("ONNX Models|*.onnx|All Files|*.*")]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("Settings")]
        public OnnxProvider Provider { get; set; } = OnnxProvider.Cuda;

        [Category("Settings")]
        public int DeviceId { get; set; } = 0;

        [Category("Dimensions")]
        public int TimePoints { get; set; } = 92;

        [Category("Dimensions")]
        public int Channels { get; set; } = 8;
        /* ───────────────────────────────────────────────────────────────── */

        // Resources
        private InferenceSession _session;
        private OrtIoBinding _binding;
        private RunOptions _runOptions;
        private OrtMemoryInfo _memInfo;
        private Stopwatch _timer = new Stopwatch();

        // Buffers
        private float[] _inputBuffer;
        private GCHandle _inputPin;
        private OrtValue _inputOrtValue;

        private float[] _outputBuffer;
        private GCHandle _outputPin;
        private OrtValue _outputOrtValue;

        // Caching for Zero-Allocation
        private int _currentCapacity = 0;
        private int _activeBatchSize = 0;
        private string _inputName, _outputName;

        // 1. Reusable Output Mat to prevent 5-10ms GC spikes
        private Mat _cachedOutputMat;

        // 2. Reusable List wrapper to prevent array allocation on single inputs
        private readonly List<Mat> _singleInputBatch = new List<Mat>(1);

        private void InitializeSession()
        {
            if (_session != null) return;

            // 1. PERFORMANCE CORE OPTIMIZATION (Safe Priority)
            // Request High priority to target P-Cores, but avoid RealTime to prevent lockups.
            try
            {
                using (var proc = System.Diagnostics.Process.GetCurrentProcess())
                {
                    if (proc.PriorityClass != System.Diagnostics.ProcessPriorityClass.High)
                    {
                        proc.PriorityClass = System.Diagnostics.ProcessPriorityClass.High;
                    }
                }
            }
            catch { /* Ignore permissions errors if run as non-admin */ }

            // 2. ANTI-FREEZE THREADING
            // Limiting to 1 thread while on High Priority effectively locks this to 
            // a single P-Core, leaving the rest of the CPU free for Windows/UI.
            var opts = new SessionOptions
            {
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
            };

            try
            {
                if (Provider == OnnxProvider.TensorRT)
                {
                    Environment.SetEnvironmentVariable("ORT_TENSORRT_FP16_ENABLE", "1");
                    opts.AppendExecutionProvider_Tensorrt(DeviceId);
                    opts.AppendExecutionProvider_CUDA(DeviceId);
                }
                else if (Provider == OnnxProvider.Cuda)
                {
                    opts.AppendExecutionProvider_CUDA(DeviceId);
                }
            }
            catch
            {
                Provider = OnnxProvider.Cpu;
            }

            _session = new InferenceSession(ModelPath, opts);
            _memInfo = OrtMemoryInfo.DefaultInstance;
            _binding = _session.CreateIoBinding();
            _runOptions = new RunOptions();
            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();
            opts.Dispose();
        }

        private void PrepareBinding(int batchSize)
        {
            InitializeSession();

            // Resize buffers if needed
            if (batchSize > _currentCapacity)
            {
                if (_inputPin.IsAllocated) _inputPin.Free();
                if (_outputPin.IsAllocated) _outputPin.Free();

                _currentCapacity = Math.Max(batchSize, _currentCapacity);

                _inputBuffer = new float[_currentCapacity * TimePoints * Channels];
                _inputPin = GCHandle.Alloc(_inputBuffer, GCHandleType.Pinned);

                _outputBuffer = new float[_currentCapacity];
                _outputPin = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);

                // Reset Cached Output Mat since capacity changed
                _cachedOutputMat = null;
            }

            // Re-bind tensors if batch size changes or capacity grew
            if (batchSize != _activeBatchSize || batchSize > _currentCapacity)
            {
                _inputOrtValue?.Dispose();
                _outputOrtValue?.Dispose();

                unsafe
                {
                    long[] inShape = { batchSize, TimePoints, Channels };
                    long[] outShape = { batchSize, 1 };

                    _inputOrtValue = OrtValue.CreateTensorValueWithData(
                        _memInfo,
                        TensorElementType.Float,
                        inShape,
                        _inputPin.AddrOfPinnedObject(),
                        batchSize * TimePoints * Channels * sizeof(float));

                    _outputOrtValue = OrtValue.CreateTensorValueWithData(
                        _memInfo,
                        TensorElementType.Float,
                        outShape,
                        _outputPin.AddrOfPinnedObject(),
                        batchSize * sizeof(float));
                }

                _binding.ClearBoundInputs();
                _binding.BindInput(_inputName, _inputOrtValue);
                _binding.ClearBoundOutputs();
                _binding.BindOutput(_outputName, _outputOrtValue);

                _activeBatchSize = batchSize;

                // Create/Resize the Reusable Output Mat ONCE
                _cachedOutputMat = new Mat(batchSize, 1, Depth.F32, 1);
            }
        }

        // Optimized Inference Method
        private PredictionResult RunInference(IList<Mat> mats)
        {
            int batchSize = mats.Count;
            if (batchSize == 0) return (null, 0);

            PrepareBinding(batchSize);

            // 1. Copy Data (Unsafe Pointer Copy is fastest)
            unsafe
            {
                float* dstBase = (float*)_inputPin.AddrOfPinnedObject().ToPointer();
                int stride = TimePoints * Channels;
                long bytesPerMat = stride * sizeof(float);

                for (int i = 0; i < batchSize; i++)
                {
                    float* src = (float*)mats[i].Data.ToPointer();
                    Buffer.MemoryCopy(src, dstBase + (i * stride), bytesPerMat, bytesPerMat);
                }
            }

            // 2. Run
            _timer.Restart();
            _session.RunWithBinding(_runOptions, _binding);
            _timer.Stop();
            double duration = _timer.Elapsed.TotalMilliseconds;

            // 3. Copy Result to Cached Mat
            unsafe
            {
                float* resultPtr = (float*)_cachedOutputMat.Data.ToPointer();
                Marshal.Copy(_outputBuffer, 0, (IntPtr)resultPtr, batchSize);
            }

            // Return struct (Stack allocation only)
            return (_cachedOutputMat, duration);
        }

        // Optimized Single-Input Path
        public IObservable<PredictionResult> Process(IObservable<Mat> source)
        {
            return Observable.Using(
                () => this,
                resource => source.Select(m =>
                {
                    // Reuse the list wrapper to avoid allocating "new[] { m }"
                    resource._singleInputBatch.Clear();
                    resource._singleInputBatch.Add(m);
                    return resource.RunInference(resource._singleInputBatch);
                })
            );
        }

        public IObservable<PredictionResult> Process(IObservable<IList<Mat>> source)
        {
            return Observable.Using(
               () => this,
               resource => source.Select(batch => resource.RunInference(batch))
           );
        }

        public void Dispose()
        {
            _inputOrtValue?.Dispose();
            _outputOrtValue?.Dispose();

            if (_inputPin.IsAllocated) _inputPin.Free();
            if (_outputPin.IsAllocated) _outputPin.Free();

            _binding?.Dispose();
            _session?.Dispose();
            _runOptions?.Dispose();
            _session = null;

            _cachedOutputMat = null;
            _singleInputBatch.Clear();
        }
    }
}