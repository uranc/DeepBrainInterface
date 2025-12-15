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

// Alias for zero-allocation return
using PredictionResult = System.ValueTuple<OpenCV.Net.Mat, double>;

namespace DeepBrainInterface
{
    public enum OnnxProvider { Cpu, Cuda, TensorRT }

    [Combinator]
    [Description("High-Performance GPU Inference. Returns (PredictionMat, DurationMs).")]
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
        [Description("Execution Provider (GPU/CPU).")]
        public OnnxProvider Provider { get; set; } = OnnxProvider.Cuda;

        [Category("Settings")]
        [Description("GPU Device ID (usually 0).")]
        public int DeviceId { get; set; } = 0;

        [Category("Settings")]
        [Description("Intra-Op Threads: Parallelizes single operators. 1 is best for low latency.")]
        public int IntraOpNumThreads { get; set; } = 1;

        [Category("Settings")]
        [Description("Inter-Op Threads: Runs graph nodes in parallel. Keep 1 for sequential.")]
        public int InterOpNumThreads { get; set; } = 1;

        [Category("Optimizations")]
        [Description("Forces execution on Cores 0-7 (P-Cores). Fixes latency on Intel 12th/13th/14th Gen Hybrid CPUs.")]
        public bool PinToPCores { get; set; } = false;

        [Category("Optimizations")]
        [Description("Allows the CPU to busy-wait (spin) for tasks. Reduces latency but increases CPU usage.")]
        public bool AllowSpinning { get; set; } = true;

        [Category("Dimensions")]
        [Description("Number of samples to process in parallel.")]
        public int BatchSize { get; set; } = 2;

        [Category("Dimensions")]
        public int TimePoints { get; set; } = 92;

        [Category("Dimensions")]
        public int Channels { get; set; } = 8;
        /* ───────────────────────────────────────────────────────────────── */

        // Resources
        private InferenceSession _session;
        private string _inputName;
        private Stopwatch _timer = new Stopwatch();

        // DenseTensor buffers
        private float[] _buffer;
        private GCHandle _bufferPin;
        private List<NamedOnnxValue> _inputs = new List<NamedOnnxValue>(1);

        // Dimensions
        private int _inputStride;
        private int _activeBatchSize;

        // Input cache
        private readonly List<Mat> _inputBatch = new List<Mat>(4);

        private void InitializeSession(int batchSize)
        {
            if (_session != null && _activeBatchSize == batchSize) return;
            if (_session != null) Dispose();

            _activeBatchSize = batchSize;
            _inputStride = TimePoints * Channels;

            // 1. CPU & THREAD OPTIMIZATIONS
            try
            {
                using (var proc = System.Diagnostics.Process.GetCurrentProcess())
                {
                    // A. High Priority
                    if (proc.PriorityClass != System.Diagnostics.ProcessPriorityClass.High)
                        proc.PriorityClass = System.Diagnostics.ProcessPriorityClass.High;

                    // B. P-Core Pinning
                    if (PinToPCores)
                    {
                        long affinityMask = (long)proc.ProcessorAffinity;
                        long pCoreMask = 0xFF;

                        if ((affinityMask & pCoreMask) == pCoreMask)
                        {
                            proc.ProcessorAffinity = (IntPtr)pCoreMask;
                        }
                    }
                }
            }
            catch { }

            // 2. Allocate Buffer
            _buffer = new float[_activeBatchSize * _inputStride];
            _bufferPin = GCHandle.Alloc(_buffer, GCHandleType.Pinned);

            // 3. Configure Session Options
            var opts = new SessionOptions
            {
                IntraOpNumThreads = IntraOpNumThreads,
                InterOpNumThreads = InterOpNumThreads,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                EnableCpuMemArena = true
            };

            // Critical Parity Settings
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            opts.AddSessionConfigEntry("session.enable_mem_pattern", "1");
            opts.AddSessionConfigEntry("session.execution_mode", "ORT_SEQUENTIAL");
            opts.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested"); // Fixed typo 'options' -> 'opts'

            // Latency Optimization (Spinning)
            if (AllowSpinning)
            {
                opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");
            }
            else
            {
                opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
            }

            // 4. Set Provider
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
            catch { Provider = OnnxProvider.Cpu; }

            _session = new InferenceSession(ModelPath, opts);
            _inputName = _session.InputMetadata.Keys.First();
            opts.Dispose();

            // Warmup
            var warmup = new DenseTensor<float>(_buffer, new[] { _activeBatchSize, TimePoints, Channels });
            using (var _ = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, warmup) })) { }
        }

        private DenseTensor<float> BuildTensor(IList<Mat> mats)
        {
            unsafe
            {
                float* dstBase = (float*)_bufferPin.AddrOfPinnedObject().ToPointer();

                for (int i = 0; i < mats.Count; i++)
                {
                    float* src = (float*)mats[i].Data.ToPointer();
                    float* dst = dstBase + (i * _inputStride);

                    // CASE A: Perfect Match [Time x Channel]
                    if (mats[i].Rows == TimePoints && mats[i].Cols == Channels)
                    {
                        Buffer.MemoryCopy(src, dst, _inputStride * sizeof(float), _inputStride * sizeof(float));
                    }
                    // CASE B: Transpose Needed [Channel x Time]
                    else if (mats[i].Rows == Channels && mats[i].Cols == TimePoints)
                    {
                        for (int c = 0; c < Channels; c++)
                            for (int t = 0; t < TimePoints; t++)
                                dst[(t * Channels) + c] = src[(c * TimePoints) + t];
                    }
                    else
                    {
                        throw new ArgumentException($"Invalid Mat Shape: {mats[i].Rows}x{mats[i].Cols}. Expected {TimePoints}x{Channels}.");
                    }
                }
            }
            return new DenseTensor<float>(_buffer, new[] { _activeBatchSize, TimePoints, Channels });
        }

        private PredictionResult RunInference(IList<Mat> mats)
        {
            int currentBatch = mats.Count;
            if (currentBatch == 0) return (null, 0);

            InitializeSession(currentBatch);

            var tensor = BuildTensor(mats);
            var named = NamedOnnxValue.CreateFromTensor(_inputName, tensor);

            _inputs.Clear();
            _inputs.Add(named);

            _timer.Restart();

            // Output Allocation
            var resultMat = new Mat(1, 1, Depth.F32, currentBatch);

            using (var results = _session.Run(_inputs))
            {
                _timer.Stop();
                var outTensor = results.First().AsTensor<float>();

                unsafe
                {
                    float* dst = (float*)resultMat.Data.ToPointer();
                    for (int b = 0; b < currentBatch; b++)
                    {
                        dst[b] = outTensor.GetValue(b);
                    }
                }
            }

            return (resultMat, _timer.Elapsed.TotalMilliseconds);
        }

        // ==============================================================================
        // OVERLOADS
        // ==============================================================================

        public IObservable<PredictionResult> Process(IObservable<Mat> source)
        {
            return Observable.Using(
                () => this,
                resource => source.Select(m =>
                {
                    resource._inputBatch.Clear();
                    resource._inputBatch.Add(m);
                    return resource.RunInference(resource._inputBatch);
                })
            );
        }

        public IObservable<PredictionResult> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return Observable.Using(
                () => this,
                resource => source.Select(t =>
                {
                    resource._inputBatch.Clear();
                    resource._inputBatch.Add(t.Item1);
                    resource._inputBatch.Add(t.Item2);
                    return resource.RunInference(resource._inputBatch);
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
            if (_bufferPin.IsAllocated) _bufferPin.Free();
            _session?.Dispose();
            _session = null;
            _inputBatch.Clear();
        }
    }
}