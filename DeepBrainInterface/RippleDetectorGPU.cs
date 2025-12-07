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
    [Description("High-Performance GPU Inference (Strict Batch 1 or 2, Pinned Memory, Tunable Threading).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorGPU
    {
        // ==============================================================================
        // 1. CONFIGURATION
        // ==============================================================================
        [Category("Model")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("Model")]
        [Description("Strictly 1 (Signal only) or 2 (Signal + Artifact).")]
        public int BatchSize { get; set; } = 2;

        [Category("Model")]
        public int TimePoints { get; set; } = 92;

        [Category("Model")]
        public int Channels { get; set; } = 8;

        // ==============================================================================
        // 2. GPU & THREADING
        // ==============================================================================
        [Category("Execution Provider")]
        [Description("Select Hardware Accelerator.")]
        public OnnxProvider Provider { get; set; } = OnnxProvider.Cuda;

        [Category("Execution Provider")]
        [Description("GPU Device ID (Default 0).")]
        public int DeviceId { get; set; } = 0;

        [Category("Threading")]
        [Description("Threads for single-op calculation (MatMul). Increase for large batches.")]
        public int IntraOpNumThreads { get; set; } = 1;

        [Category("Threading")]
        [Description("Threads for parallel sub-graphs.")]
        public int InterOpNumThreads { get; set; } = 1;

        // ==============================================================================
        // INTERNAL RESOURCES
        // ==============================================================================
        InferenceSession _session;
        OrtIoBinding _ioBinding;
        RunOptions _runOptions;
        OrtMemoryInfo _memInfo;

        // Fixed Pinned Buffers
        float[] _inputBuffer;
        GCHandle _inputPin;
        OrtValue _inputOrtValue;

        float[] _outputBuffer;
        GCHandle _outputPin;
        OrtValue _outputOrtValue;

        struct InputPackage { public Mat[] Mats; }

        private void Initialise()
        {
            if (_session != null) return;
            if (!File.Exists(ModelPath)) throw new FileNotFoundException("Model not found", ModelPath);

            // 1. CONFIGURE SESSION OPTIONS
            var opts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                IntraOpNumThreads = IntraOpNumThreads,
                InterOpNumThreads = InterOpNumThreads
            };

            // 2. CONFIGURE GPU PROVIDERS (Robust Fallback Logic)
            try
            {
                if (Provider == OnnxProvider.TensorRT)
                {
                    // TensorRT Setup via Environment Variables (Fixes API compatibility)
                    string cacheDir = Path.GetDirectoryName(ModelPath);
                    Environment.SetEnvironmentVariable("ORT_TENSORRT_FP16_ENABLE", "1");
                    Environment.SetEnvironmentVariable("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1");
                    Environment.SetEnvironmentVariable("ORT_TENSORRT_ENGINE_CACHE_PATH", cacheDir);

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
                Console.WriteLine($"[GPU Error] {ex.Message}. Falling back to CPU.");
                Provider = OnnxProvider.Cpu;
            }

            // 3. CREATE SESSION
            _session = new InferenceSession(ModelPath, opts);

            // NOTE: Even for GPU, we use CPU memory for the binding source. 
            // ONNX Runtime handles the DMA transfer to GPU automatically.
            _memInfo = OrtMemoryInfo.DefaultInstance;

            _ioBinding = _session.CreateIoBinding();
            _runOptions = new RunOptions();

            // 4. STRICT MEMORY ALLOCATION
            int inputLen = BatchSize * TimePoints * Channels;
            _inputBuffer = new float[inputLen];
            _inputPin = GCHandle.Alloc(_inputBuffer, GCHandleType.Pinned);

            int outputLen = BatchSize; // [Batch, 1]
            _outputBuffer = new float[outputLen];
            _outputPin = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);

            // 5. BINDING SETUP
            var inputName = _session.InputMetadata.Keys.First();
            var outputName = _session.OutputMetadata.Keys.First();

            long[] inShape = new long[] { BatchSize, TimePoints, Channels };
            long[] outShape = new long[] { BatchSize, 1 };

            unsafe
            {
                _inputOrtValue = OrtValue.CreateTensorValueWithData(
                    _memInfo,
                    TensorElementType.Float,
                    inShape,
                    _inputPin.AddrOfPinnedObject(),
                    inputLen * sizeof(float)
                );

                _outputOrtValue = OrtValue.CreateTensorValueWithData(
                    _memInfo,
                    TensorElementType.Float,
                    outShape,
                    _outputPin.AddrOfPinnedObject(),
                    outputLen * sizeof(float)
                );
            }

            _ioBinding.BindInput(inputName, _inputOrtValue);
            _ioBinding.BindOutput(outputName, _outputOrtValue);

            // Warmup Run
            _session.RunWithBinding(_runOptions, _ioBinding);
        }

        private IObservable<Mat> ProcessInternal(IObservable<InputPackage> source)
        {
            return source.Select(input =>
            {
                Initialise();

                // Safety Check
                if (input.Mats.Length != BatchSize)
                    throw new InvalidOperationException($"Input count {input.Mats.Length} does not match BatchSize {BatchSize}");

                // 1. FAST COPY (Mat -> Pinned Buffer)
                unsafe
                {
                    float* dstBase = (float*)_inputPin.AddrOfPinnedObject().ToPointer();
                    int stride = TimePoints * Channels;
                    long bytesPerMat = stride * sizeof(float);

                    for (int i = 0; i < BatchSize; i++)
                    {
                        float* src = (float*)input.Mats[i].Data.ToPointer();
                        float* dst = dstBase + (i * stride);

                        // Layout Check
                        if (input.Mats[i].Rows == TimePoints && input.Mats[i].Cols == Channels)
                        {
                            Buffer.MemoryCopy(src, dst, bytesPerMat, bytesPerMat);
                        }
                        else if (input.Mats[i].Rows == Channels && input.Mats[i].Cols == TimePoints)
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

                // 2. INFERENCE
                // Data transfers from Pinned CPU RAM -> GPU VRAM -> Compute -> GPU VRAM -> Pinned CPU RAM
                // This happens automatically inside RunWithBinding
                _session.RunWithBinding(_runOptions, _ioBinding);

                // 3. RETURN RESULT
                var outMat = new Mat(BatchSize, 1, Depth.F32, 1);
                unsafe
                {
                    float* dst = (float*)outMat.Data.ToPointer();
                    Marshal.Copy(_outputBuffer, 0, (IntPtr)dst, BatchSize);
                }

                return outMat;
            });
        }

        // --- PIPELINE OVERLOADS ---

        public IObservable<Mat> Process(IObservable<Mat> source)
            => ProcessInternal(source.Select(m => new InputPackage { Mats = new[] { m } }));

        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
            => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1, t.Item2 } }));

        public void Unload()
        {
            _inputOrtValue?.Dispose(); _outputOrtValue?.Dispose();
            if (_inputPin.IsAllocated) _inputPin.Free();
            if (_outputPin.IsAllocated) _outputPin.Free();
            _ioBinding?.Dispose(); _runOptions?.Dispose(); _session?.Dispose();
        }
    }
}