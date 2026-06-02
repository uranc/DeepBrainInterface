using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Hyper-Lean CPU Inference. Pure Mat Output. Thread-Safe.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorCPU : IDisposable
    {
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        const uint _MCW_DN = 0x03000000;
        const uint _DN_FLUSH = 0x01000000;

        [Category("1. Model")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("2. Dimensions")] public int BatchSize { get; set; } = 1;
        [Category("2. Dimensions")] public int TimePoints { get; set; } = 44;
        [Category("2. Dimensions")] public int Channels { get; set; } = 8;

        [Category("3. Hardware")] public bool AllowSpinning { get; set; } = true;

        private readonly object _inferenceLock = new object();

        private InferenceSession _session;
        private OrtIoBinding _binding;
        private RunOptions _runOpts;

        private float[] _bufIn, _bufOut;
        private GCHandle _hIn, _hOut;
        private OrtValue _valIn, _valOut;

        private int _inputStride;
        private int _expectedBytes;
        private int _expectedContiguousStep;
        private int _outCols;
        private int _outSizeBytes;

        private readonly List<Mat> _inputCache = new List<Mat>(2);

        private void InitializeSession()
        {
            if (_session != null) return;

            _inputStride = TimePoints * Channels;
            _expectedBytes = _inputStride * sizeof(float);
            _expectedContiguousStep = Channels * sizeof(float);

            // 1. Hardware Subnormal Float Flush 
            try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { }

            // 2. Process Escalation
            try
            {
                using (var process = System.Diagnostics.Process.GetCurrentProcess())
                    process.PriorityClass = System.Diagnostics.ProcessPriorityClass.RealTime;
                System.Threading.Thread.CurrentThread.Priority = System.Threading.ThreadPriority.Highest;
            }
            catch { }

            // 3. Ultra-Lean CPU ORT Config
            var opts = new SessionOptions
            {
                IntraOpNumThreads = 1, // Hardcoded for real-time determinism
                InterOpNumThreads = 1,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                EnableCpuMemArena = true,
                EnableMemoryPattern = true
            };

            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            opts.AddSessionConfigEntry("session.execution_mode", "ORT_SEQUENTIAL");
            opts.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", AllowSpinning ? "1" : "0");

            _session = new InferenceSession(ModelPath, opts);
            _binding = _session.CreateIoBinding();
            _runOpts = new RunOptions();
            opts.Dispose();

            long[] inShape = new long[] { BatchSize, TimePoints, Channels };
            int inSize = BatchSize * _inputStride;

            var outMeta = _session.OutputMetadata.Values.First();
            long[] outShape = new long[outMeta.Dimensions.Length];
            int outSize = 1;
            for (int i = 0; i < outShape.Length; i++)
            {
                long d = outMeta.Dimensions[i];
                if (d <= 0) d = (i == 0) ? BatchSize : 1;
                outShape[i] = d;
                outSize *= (int)d;
            }

            _outCols = outSize / BatchSize;
            _outSizeBytes = outSize * sizeof(float);

            // 4. Pin memory and bind directly to OrtValues 
            _bufIn = new float[inSize];
            _hIn = GCHandle.Alloc(_bufIn, GCHandleType.Pinned);

            _bufOut = new float[outSize];
            _hOut = GCHandle.Alloc(_bufOut, GCHandleType.Pinned);

            var memInfo = OrtMemoryInfo.DefaultInstance;
            _valIn = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufIn), inShape);
            _valOut = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufOut), outShape);

            _binding.BindInput(_session.InputMetadata.Keys.First(), _valIn);
            _binding.BindOutput(_session.OutputMetadata.Keys.First(), _valOut);

            // 5. Deep Warmup
            for (int i = 0; i < 50; i++)
            {
                _session.RunWithBinding(_runOpts, _binding);
            }
        }

        private Mat RunInference(IList<Mat> mats)
        {
            lock (_inferenceLock)
            {
                if (_session == null) InitializeSession();

                unsafe
                {
                    float* dstBase = (float*)_hIn.AddrOfPinnedObject().ToPointer();

                    for (int i = 0; i < mats.Count; i++)
                    {
                        Mat inputMat = mats[i];
                        byte* srcBase = (byte*)inputMat.Data.ToPointer();
                        float* dst = dstBase + (i * _inputStride);

                        if (inputMat.Rows == TimePoints && inputMat.Cols == Channels && inputMat.Step == _expectedContiguousStep)
                        {
                            Buffer.MemoryCopy(srcBase, dst, _expectedBytes, _expectedBytes);
                        }
                        else
                        {
                            int step = inputMat.Step;
                            if (inputMat.Rows == Channels && inputMat.Cols == TimePoints)
                            {
                                for (int c = 0; c < Channels; c++)
                                {
                                    float* srcRow = (float*)(srcBase + (c * step));
                                    for (int t = 0; t < TimePoints; t++)
                                        dst[(t * Channels) + c] = srcRow[t];
                                }
                            }
                            else
                            {
                                int cols = inputMat.Cols;
                                int rowBytes = cols * sizeof(float);
                                for (int r = 0; r < inputMat.Rows; r++)
                                {
                                    Buffer.MemoryCopy(srcBase + (r * step), (byte*)(dst + (r * cols)), rowBytes, rowBytes);
                                }
                            }
                        }
                    }
                }

                _session.RunWithBinding(_runOpts, _binding);

                // Export clean, downstream-safe native Mat
                var resultMat = new Mat(BatchSize, _outCols, Depth.F32, 1);
                unsafe
                {
                    Buffer.MemoryCopy(
                        _hOut.AddrOfPinnedObject().ToPointer(),
                        resultMat.Data.ToPointer(),
                        _outSizeBytes,
                        _outSizeBytes);
                }

                return resultMat;
            }
        }

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(m =>
            {
                lock (_inferenceLock)
                {
                    _inputCache.Clear();
                    _inputCache.Add(m);
                    return RunInference(_inputCache);
                }
            });
        }

        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return source.Select(t =>
            {
                lock (_inferenceLock)
                {
                    _inputCache.Clear();
                    _inputCache.Add(t.Item1);
                    _inputCache.Add(t.Item2);
                    return RunInference(_inputCache);
                }
            });
        }

        public IObservable<Mat> Process(IObservable<IList<Mat>> source)
        {
            return source.Select(batch => RunInference(batch));
        }

        private void ReleaseAllocations()
        {
            if (_hIn.IsAllocated) _hIn.Free();
            if (_hOut.IsAllocated) _hOut.Free();

            _valIn?.Dispose();
            _valOut?.Dispose();
            _binding?.Dispose();
            _runOpts?.Dispose();
            _session?.Dispose();

            _valIn = null;
            _valOut = null;
            _binding = null;
            _runOpts = null;
            _session = null;
        }

        public void Dispose()
        {
            lock (_inferenceLock)
            {
                ReleaseAllocations();
                _inputCache.Clear();
            }
        }
    }
}