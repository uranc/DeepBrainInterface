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
using System.Diagnostics;
using System.Runtime;

namespace DeepBrainInterface
{
    public enum OnnxProvider { Cpu, Cuda, TensorRT }

    [Combinator]
    [Description("High-Performance GPU Inference (Zero-Allocation Loop).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorGPU
    {
        // ... (Configuration properties same as before) ...
        [Category("Model")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("Model")]
        public int BatchSize { get; set; } = 2;

        [Category("Data Dimensions")]
        public int TimePoints { get; set; } = 44;

        [Category("Data Dimensions")]
        public int Channels { get; set; } = 8;

        [Category("Execution Provider")]
        public OnnxProvider Provider { get; set; } = OnnxProvider.Cuda;

        [Category("Execution Provider")]
        public int DeviceId { get; set; } = 0;

        [Category("Threading")]
        public int IntraOpNumThreads { get; set; } = 1;

        [Category("Threading")]
        public int InterOpNumThreads { get; set; } = 1;

        // ... Resources ...
        private InferenceSession _session;
        private OrtIoBinding _ioBinding;
        private RunOptions _runOptions;
        private Stopwatch _benchmarker = new Stopwatch();

        // Memory Resources
        private GCHandle _inputPin;
        private GCHandle _outputPin;
        private OrtValue _inputOrtValue;
        private OrtValue _outputOrtValue;
        private float[] _outputBuffer; // The raw pinned memory for ONNX

        // NEW: The reusable container for Bonsai
        private Mat _reusableOutputMat;

        private int _batchStrideFloats;

        // Optimization: Track GC to prove spikes
        private int _lastGcCount = 0;

        private void Initialise()
        {
            if (_session != null) return;

            // 1. System Priority (Crucial for <1ms stability)
            try
            {
                Process.GetCurrentProcess().PriorityClass = ProcessPriorityClass.RealTime;
                GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;
            }
            catch (Exception ex) { Console.WriteLine(ex.Message); }

            // 2. ONNX Config
            var opts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                IntraOpNumThreads = IntraOpNumThreads,
                InterOpNumThreads = InterOpNumThreads
            };

            if (Provider == OnnxProvider.TensorRT)
            {
                string cacheDir = Path.GetDirectoryName(Path.GetFullPath(ModelPath));
                Environment.SetEnvironmentVariable("ORT_TENSORRT_FP16_ENABLE", "1");
                Environment.SetEnvironmentVariable("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1");
                Environment.SetEnvironmentVariable("ORT_TENSORRT_ENGINE_CACHE_PATH", cacheDir);
            }

            try
            {
                if (Provider == OnnxProvider.TensorRT)
                {
                    opts.AppendExecutionProvider_Tensorrt(DeviceId);
                    opts.AppendExecutionProvider_CUDA(DeviceId);
                }
                else if (Provider == OnnxProvider.Cuda) opts.AppendExecutionProvider_CUDA(DeviceId);
            }
            catch (Exception ex) { Console.WriteLine($"[GPU Init Failed] {ex.Message}"); }

            _session = new InferenceSession(ModelPath, opts);
            _ioBinding = _session.CreateIoBinding();
            _runOptions = new RunOptions();

            // 3. Allocations (DONE EXACTLY ONCE)
            _batchStrideFloats = TimePoints * Channels;
            int totalInputFloats = BatchSize * _batchStrideFloats;

            var inputData = new float[totalInputFloats];
            _inputPin = GCHandle.Alloc(inputData, GCHandleType.Pinned);

            _outputBuffer = new float[BatchSize];
            _outputPin = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);

            // PRE-ALLOCATE THE OUTPUT MAT
            // We will never create a new Mat() after this line.
            _reusableOutputMat = new Mat(BatchSize, 1, Depth.F32, 1);

            // Bindings
            var memInfo = OrtMemoryInfo.DefaultInstance;
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

        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return source.Select(input =>
            {
                Initialise();
                unsafe
                {
                    float* ptr = (float*)_inputPin.AddrOfPinnedObject();
                    RobustTransposeCopy(input.Item1, ptr);
                    RobustTransposeCopy(input.Item2, ptr + _batchStrideFloats);
                }
                return RunInference();
            });
        }

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                Initialise();
                unsafe
                {
                    RobustTransposeCopy(input, (float*)_inputPin.AddrOfPinnedObject());
                }
                return RunInference();
            });
        }

        private unsafe void RobustTransposeCopy(Mat src, float* dstBase)
        {
            byte* srcRowPtr = (byte*)src.Data.ToPointer();
            int srcStep = src.Step;
            for (int c = 0; c < Channels; c++)
            {
                float* srcFloatRow = (float*)srcRowPtr;
                for (int t = 0; t < TimePoints; t++)
                {
                    dstBase[t * Channels + c] = srcFloatRow[t];
                }
                srcRowPtr += srcStep;
            }
        }

        private Mat RunInference()
        {
            // 1. Snapshot GC State
            int currentGcCount = GC.CollectionCount(0);
            bool gcHappened = currentGcCount > _lastGcCount;
            _lastGcCount = currentGcCount;

            _benchmarker.Restart();

            // 2. Run ONNX (Writes directly to _outputBuffer via IoBinding)
            _session.RunWithBinding(_runOptions, _ioBinding);

            // 3. Update the Reusable Mat
            // Use Marshal.Copy to move data from the pinned float[] to the Mat's internal pointer.
            // This is a memory-to-memory copy of ~1KB (negligible time) but avoids object creation.
            unsafe
            {
                Marshal.Copy(_outputBuffer, 0, _reusableOutputMat.Data, BatchSize);
            }

            _benchmarker.Stop();

            // 4. Smart Logging (Only print spikes)
            if (gcHappened || _benchmarker.Elapsed.TotalMilliseconds > 1.0)
            {
                string log = $"SPIKE: {_benchmarker.Elapsed.TotalMilliseconds:F4} ms";
                if (gcHappened) log += " <--- *** GARBAGE COLLECTION ***";
                Console.WriteLine(log);
            }

            return _reusableOutputMat;
        }

        public void Unload()
        {
            _reusableOutputMat?.Dispose();
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