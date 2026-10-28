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

        [Browsable(false)] // no effect with IntraOpNumThreads = 1 (no thread pool); kept so old workflow files load
        [Category("3. Hardware")] public bool AllowSpinning { get; set; } = true;

        [Category("3. Hardware")]
        [Description("Legacy behaviour: raise the process to RealTime and the inference thread to Highest from inside this node. " +
                     "Set false when a RealtimeConfig node manages priorities for the whole workflow.")]
        public bool ManagePriority { get; set; } = true;

        [Category("3. Hardware")]
        [Description("Number of pre-allocated output Mats cycled through instead of allocating one per inference. " +
                     "Downstream consumers must not hold an emitted Mat for more than this many emissions (256 = ~300 ms at 833 Hz).")]
        public int OutputRingDepth { get; set; } = 256;

        private readonly object _inferenceLock = new object();
        private Mat[] _outRing;
        private int _outIdx;
        private bool _hotThreadBoosted;

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

            // 2. Process Escalation (legacy; prefer the RealtimeConfig node and ManagePriority = false).
            //    The thread boost is applied lazily on the first inference so it lands on the data thread,
            //    not on whichever thread subscribes.
            if (ManagePriority)
            {
                try
                {
                    using (var process = System.Diagnostics.Process.GetCurrentProcess())
                        process.PriorityClass = System.Diagnostics.ProcessPriorityClass.RealTime;
                }
                catch { }
            }

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

            // Output ring: allocated once, cycled per inference (no per-call Mat + finalizer churn)
            int ringDepth = Math.Max(2, OutputRingDepth);
            _outRing = new Mat[ringDepth];
            for (int i = 0; i < ringDepth; i++) _outRing[i] = new Mat(BatchSize, _outCols, Depth.F32, 1);
            _outIdx = 0;

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

                if (ManagePriority && !_hotThreadBoosted)
                {
                    try { System.Threading.Thread.CurrentThread.Priority = System.Threading.ThreadPriority.Highest; } catch { }
                    _hotThreadBoosted = true;
                }

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

                // Export via the pre-allocated output ring (each emission is a distinct Mat; recycled after OutputRingDepth emissions)
                var resultMat = _outRing[_outIdx];
                _outIdx = (_outIdx + 1 == _outRing.Length) ? 0 : _outIdx + 1;
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

        // Session creation + warm-up run at subscribe time (workflow start), not on the first data frame
        // on the acquisition thread. The guard inside RunInference stays as a safety net.
        private void EnsureInitialized()
        {
            lock (_inferenceLock)
            {
                if (_session == null) InitializeSession();
            }
        }

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Defer(() =>
            {
                EnsureInitialized();
                return source.Select(m =>
                {
                    lock (_inferenceLock)
                    {
                        _inputCache.Clear();
                        _inputCache.Add(m);
                        return RunInference(_inputCache);
                    }
                });
            });
        }

        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return Observable.Defer(() =>
            {
                EnsureInitialized();
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
            });
        }

        public IObservable<Mat> Process(IObservable<IList<Mat>> source)
        {
            return Observable.Defer(() =>
            {
                EnsureInitialized();
                return source.Select(batch => RunInference(batch));
            });
        }

        // Clock pass-through: Zip(Buffer(44,3), Rhd2164.Clock) -> RippleDetectorCPU
        // -> Tuple<probability Mat, ulong clock> for downstream RippleStateMachineMatBool or SuperNode.
        // Scalar ulong: single clock per window.
        public IObservable<Tuple<Mat, ulong>> Process(IObservable<Tuple<Mat, ulong>> source)
        {
            return Observable.Defer(() =>
            {
                EnsureInitialized();
                return source.Select(t =>
                {
                    lock (_inferenceLock)
                    {
                        _inputCache.Clear();
                        _inputCache.Add(t.Item1);
                        return Tuple.Create(RunInference(_inputCache), t.Item2);
                    }
                });
            });
        }

        // Clock array pass-through: Rhd2164.Clock is ulong[] of length TimePoints.
        // Extract the most recent (last) clock value.
        public IObservable<Tuple<Mat, ulong>> Process(IObservable<Tuple<Mat, ulong[]>> source)
        {
            return Observable.Defer(() =>
            {
                EnsureInitialized();
                return source.Select(t =>
                {
                    lock (_inferenceLock)
                    {
                        _inputCache.Clear();
                        _inputCache.Add(t.Item1);
                        ulong clock = (t.Item2 != null && t.Item2.Length > 0) ? t.Item2[t.Item2.Length - 1] : 0UL;
                        return Tuple.Create(RunInference(_inputCache), clock);
                    }
                });
            });
        }

        private void ReleaseAllocations()
        {
            // Release ORT objects first, then unpin the buffers they referenced
            _binding?.Dispose();
            _valIn?.Dispose();
            _valOut?.Dispose();
            _runOpts?.Dispose();
            _session?.Dispose();

            if (_hIn.IsAllocated) _hIn.Free();
            if (_hOut.IsAllocated) _hOut.Free();

            if (_outRing != null)
            {
                for (int i = 0; i < _outRing.Length; i++) _outRing[i]?.Dispose();
                _outRing = null;
            }

            _valIn = null;
            _valOut = null;
            _binding = null;
            _runOpts = null;
            _session = null;
            _hotThreadBoosted = false;
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