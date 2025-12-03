using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design; // Required for [Editor]
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms.Design; // Required for [Editor]

namespace DeepBrainInterface
{
    public enum OnnxProvider { Cpu, Cuda, TensorRT }

    [Combinator]
    [Description("Ultra-Low Latency ONNX Inference (IoBinding + Full Thread Tuning).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorGPU
    {
        /* ───── User Parameters ─────────────────────────────────────────── */

        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector.onnx";

        [Description("Execution Provider.")]
        public OnnxProvider Provider { get; set; } = OnnxProvider.Cpu;

        [Description("Threads for single-op calculation (MatMul). Sweet spot usually 1-4.")]
        public int IntraOpNumThreads { get; set; } = 2;

        [Description("Threads for parallel sub-graphs. Keep at 1 for sequential models.")]
        public int InterOpNumThreads { get; set; } = 1;

        public int BatchSize { get; set; } = 1;

        [Description("Must match input size (e.g. 44 or 281).")]
        public int TimePoints { get; set; } = 44;

        public int Channels { get; set; } = 8;
        /* ───────────────────────────────────────────────────────────────── */

        // ONNX Resources
        InferenceSession _session;
        OrtIoBinding _ioBinding;
        RunOptions _runOptions;

        // Wrappers
        OrtValue _inputOrtValue;
        OrtValue _outputOrtValue;

        // Buffers
        float[] _inputBuffer;
        float[] _outputBuffer;

        // Diagnostics
        static readonly Stopwatch swRun = new Stopwatch();
        const int kPrint = 2000;
        int nAccum;
        double accRun;

        private void Initialise()
        {
            if (_session != null) return;
            if (!File.Exists(ModelPath)) throw new FileNotFoundException("Model not found", ModelPath);

            Console.WriteLine($"[Init] Configuring {Provider} | Intra: {IntraOpNumThreads} | Inter: {InterOpNumThreads}...");

            // 1. Setup Input Buffer
            _inputBuffer = new float[BatchSize * TimePoints * Channels];

            // 2. Environment Variables (TRT)
            if (Provider == OnnxProvider.TensorRT)
            {
                Environment.SetEnvironmentVariable("ORT_TENSORRT_FP16_ENABLE", "1");
                Environment.SetEnvironmentVariable("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1");
                Environment.SetEnvironmentVariable("ORT_TENSORRT_ENGINE_CACHE_PATH", Path.GetDirectoryName(ModelPath));
                Environment.SetEnvironmentVariable("ORT_TENSORRT_MAX_WORKSPACE_SIZE", "2147483648");
            }

            // 3. Session Options
            var opts = new SessionOptions
            {
                // SEQUENTIAL is mandatory to allow our manual thread tuning to take effect cleanly
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,

                // FULL USER CONTROL OVER THREADING
                IntraOpNumThreads = IntraOpNumThreads,
                InterOpNumThreads = InterOpNumThreads
            };

            try
            {
                if (Provider == OnnxProvider.TensorRT)
                {
                    var trtOpts = new OrtTensorRTProviderOptions();
                    opts.AppendExecutionProvider_Tensorrt(trtOpts);
                    opts.AppendExecutionProvider_CUDA(0);
                }
                else if (Provider == OnnxProvider.Cuda)
                {
                    opts.AppendExecutionProvider_CUDA(0);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GPU Load Failed: {ex.Message}. Fallback to CPU.");
                Provider = OnnxProvider.Cpu;
            }

            // 4. Create Session
            Console.WriteLine($"[{Provider}] Loading Model...");
            _session = new InferenceSession(ModelPath, opts);
            _runOptions = new RunOptions();

            // 5. Create IoBinding
            _ioBinding = _session.CreateIoBinding();
            var inputName = _session.InputMetadata.Keys.First();
            var outputName = _session.OutputMetadata.Keys.First();
            var memInfo = OrtMemoryInfo.DefaultInstance;

            // --- BIND INPUT ---
            long[] inputShape = new long[] { BatchSize, TimePoints, Channels };

            _inputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(
                memInfo, _inputBuffer, inputShape);
            _ioBinding.BindInput(inputName, _inputOrtValue);

            // --- BIND OUTPUT (Rank Detection) ---
            var modelDims = _session.OutputMetadata[outputName].Dimensions;
            long[] outputShape = new long[modelDims.Length];
            long totalSize = 1;

            // Detect Rank 3 vs Rank 2
            for (int i = 0; i < modelDims.Length; i++)
            {
                long dim = modelDims[i];
                if (dim < 0)
                {
                    if (i == 0) dim = BatchSize;
                    else if (modelDims.Length == 3 && i == 1) dim = TimePoints;
                    else dim = 1;
                }
                outputShape[i] = dim;
                totalSize *= dim;
            }

            _outputBuffer = new float[totalSize];
            _outputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(
                memInfo, _outputBuffer, outputShape);
            _ioBinding.BindOutput(outputName, _outputOrtValue);

            // 6. Warmup
            _session.RunWithBinding(_runOptions, _ioBinding);
            Console.WriteLine($"[{Provider}] Ready.");
        }

        private Mat ProcessBatch(IList<Mat> batch)
        {
            Initialise();

            // A. UNSAFE COPY INPUT
            unsafe
            {
                fixed (float* dstBase = _inputBuffer)
                {
                    int stride = TimePoints * Channels;
                    for (int b = 0; b < BatchSize; b++)
                    {
                        float* src = (float*)batch[b].Data.ToPointer();
                        float* dst = dstBase + (b * stride);
                        Buffer.MemoryCopy(src, dst, stride * sizeof(float), stride * sizeof(float));
                    }
                }
            }

            // B. RUN IOBINDING
            swRun.Restart();
            _session.RunWithBinding(_runOptions, _ioBinding);
            swRun.Stop();

            // C. EXTRACT OUTPUT
            var outMat = new Mat(1, BatchSize, Depth.F32, 1);
            unsafe
            {
                float* outDst = (float*)outMat.Data.ToPointer();
                fixed (float* src = _outputBuffer)
                {
                    int stride = _outputBuffer.Length / BatchSize;
                    for (int i = 0; i < BatchSize; i++)
                    {
                        outDst[i] = src[i * stride];
                    }
                }
            }

            // D. STATS
            accRun += swRun.ElapsedTicks * 1e6 / Stopwatch.Frequency;
            if (++nAccum == kPrint)
            {
                Console.WriteLine($"[{Provider} IO] Inf: {accRun / kPrint:F1} µs");
                nAccum = 0; accRun = 0;
            }

            return outMat;
        }

        public IObservable<Mat> Process(IObservable<Mat> source) => source.Select(m => ProcessBatch(new[] { m }));
        public IObservable<Mat> Process(IObservable<IList<Mat>> source) => source.Select(batch => ProcessBatch(batch));

        ~RippleDetectorGPU()
        {
            _inputOrtValue?.Dispose();
            _outputOrtValue?.Dispose();
            _ioBinding?.Dispose();
            _runOptions?.Dispose();
            _session?.Dispose();
        }
    }
}