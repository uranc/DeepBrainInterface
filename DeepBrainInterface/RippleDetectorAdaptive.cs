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
using System.Diagnostics;

namespace DeepBrainInterface
{
    public struct AdaptiveResult
    {
        public RippleOut Ripple { get; set; }
        public double ExecutionTimeMs { get; set; }
        public double InferenceTimeMs { get; set; }
    }

    [Combinator]
    [Description("Adaptive Detector. CPU FAST-PATH. Zero Allocation. Hardware-Locked. Red-Lined.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorAdaptive : IDisposable
    {
        /* ───── Windows Kernel P/Invoke Hooks ───── */
        [DllImport("kernel32.dll")]
        static extern IntPtr GetCurrentThread();

        [DllImport("kernel32.dll")]
        static extern IntPtr SetThreadAffinityMask(IntPtr hThread, IntPtr dwThreadAffinityMask);

        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);

        const uint _MCW_DN = 0x03000000;
        const uint _DN_FLUSH = 0x01000000;

        /* ───── User Properties ───── */
        [Category("1. Model")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        [FileNameFilter("ONNX Models|*.onnx|All Files|*.*")]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("2. Dimensions")] public int TimePoints { get; set; } = 44;
        [Category("2. Dimensions")] public int Channels { get; set; } = 8;

        [Category("3. System & Memory")] public int DownsampleFactor { get; set; } = 12;
        [Category("3. System & Memory")] public int RingBufferCapacity { get; set; } = 30000;

        [Category("4. Hardware & Red-Lining")]
        public int CpuCoreAffinity { get; set; } = 2;

        [Category("4. Hardware & Red-Lining")]
        public int IntraOpNumThreads { get; set; } = 1;

        [Category("4. Hardware & Red-Lining")]
        public bool AllowSpinning { get; set; } = true;

        [Category("5. Logic")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public RippleStateMachineMatBool StateMachine { get; set; } = new RippleStateMachineMatBool();

        [Category("5. Logic")] public bool DetectionEnabled { get; set; } = true;
        [Category("5. Logic")] public int KBelowGate { get; set; } = 5;
        [Category("5. Logic")] public int KAtGate { get; set; } = 1;

        /* ───── State & Buffers ───── */
        private readonly object _inferenceLock = new object();
        private bool _isThreadPinned = false;

        private InferenceSession _session;
        private RunOptions _runOpts;

        private string[] _inputNames;
        private string[] _outputNames;
        private OrtValue[] _inputValues;
        private OrtValue[] _outputValues;

        private Stopwatch _nodeTimer = new Stopwatch();
        private Stopwatch _inferenceTimer = new Stopwatch();

        private float[] _bufIn, _bufOut, _ringBuffer, _snapshotBuffer;
        private GCHandle _hIn, _hOut, _hRingBuffer, _hSnapshotBuffer;
        private Mat _snapshotMat;
        private int _outSize;

        private int _headIndex = 0;
        private int _strideCounter = 0;
        private int _currentK = 1;

        private void Initialise()
        {
            if (_session != null) return;

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

            // 3. Ultra-Lean ORT Config
            var opts = new SessionOptions
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = IntraOpNumThreads,
                InterOpNumThreads = 1,
                EnableCpuMemArena = true,
                EnableMemoryPattern = true
            };

            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            opts.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", AllowSpinning ? "1" : "0");

            _session = new InferenceSession(ModelPath, opts);
            _runOpts = new RunOptions();
            opts.Dispose();

            _inputNames = new string[] { _session.InputMetadata.Keys.First() };
            _outputNames = new string[] { _session.OutputMetadata.Keys.First() };

            int inSize = 1 * TimePoints * Channels;
            long[] inShape = new long[] { 1, TimePoints, Channels };

            var outMeta = _session.OutputMetadata.Values.First();
            long[] outShape = new long[outMeta.Dimensions.Length];
            _outSize = 1;
            for (int i = 0; i < outShape.Length; i++)
            {
                long d = outMeta.Dimensions[i];
                if (d <= 0) d = 1;
                outShape[i] = d;
                _outSize *= (int)d;
            }

            // 4. Pin memory and bind directly to OrtValues 
            _bufIn = new float[inSize];
            _hIn = GCHandle.Alloc(_bufIn, GCHandleType.Pinned);

            _bufOut = new float[_outSize];
            _hOut = GCHandle.Alloc(_bufOut, GCHandleType.Pinned);

            var memInfo = OrtMemoryInfo.DefaultInstance;
            _inputValues = new OrtValue[] { OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufIn), inShape) };
            _outputValues = new OrtValue[] { OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufOut), outShape) };

            // Data Buffers
            _ringBuffer = new float[RingBufferCapacity * Channels];
            _hRingBuffer = GCHandle.Alloc(_ringBuffer, GCHandleType.Pinned);

            _snapshotBuffer = new float[Channels * TimePoints];
            _hSnapshotBuffer = GCHandle.Alloc(_snapshotBuffer, GCHandleType.Pinned);
            _snapshotMat = new Mat(Channels, TimePoints, Depth.F32, 1, _hSnapshotBuffer.AddrOfPinnedObject());

            _currentK = KBelowGate;

            // 5. Deep Warmup
            for (int i = 0; i < 100; i++)
            {
                _session.Run(_runOpts, _inputNames, _inputValues, _outputNames, _outputValues);
            }
        }

        // --- OVERLOADS ---
        public IObservable<AdaptiveResult> Process(IObservable<Mat> source)
            => source.Select(m => ExecuteSafe(m, true)).Where(r => r.HasValue).Select(r => r.Value);

        public IObservable<AdaptiveResult> Process(IObservable<Tuple<Mat, bool>> source)
            => source.Select(t => ExecuteSafe(t.Item1, t.Item2)).Where(r => r.HasValue).Select(r => r.Value);

        public IObservable<AdaptiveResult> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
            => source.Select(t => ExecuteSafe(t.Item1.Item1, t.Item2)).Where(r => r.HasValue).Select(r => r.Value);

        private AdaptiveResult? ExecuteSafe(Mat rawData, bool isQuiet)
        {
            _nodeTimer.Restart();

            lock (_inferenceLock)
            {
                if (_session == null) Initialise();

                if (!_isThreadPinned)
                {
                    SetThreadAffinityMask(GetCurrentThread(), new IntPtr(1 << CpuCoreAffinity));
                    _isThreadPinned = true;
                }

                if (rawData.Rows != Channels)
                {
                    throw new InvalidOperationException($"FATAL SHAPE MISMATCH: Ring Buffer expects {Channels} Channels (Rows), but received {rawData.Rows} Rows.");
                }

                UpdateRingBufferUnsafe(rawData);
                PrepareInputUnsafe();

                // THE GHOST RUN: We ALWAYS run the ONNX math to keep the CPU 100% awake and L1 cache hot.
                _inferenceTimer.Restart();
                _session.Run(_runOpts, _inputNames, _inputValues, _outputNames, _outputValues);
                _inferenceTimer.Stop();

                // Downsample Gate: Drop the frame silently AFTER we did the math
                if (_currentK > 1)
                {
                    _strideCounter++;
                    if (_strideCounter < _currentK) return null;
                    _strideCounter = 0;
                }

                // Extract directly from pinned array without allocations
                float sigProb = _bufOut[0];
                float artProb = (_outSize > 1) ? _bufOut[1] : 0f;

                ExtractSnapshotUnsafe();

                StateMachine.DetectionEnabled = DetectionEnabled;
                bool gatesOpen = isQuiet && (artProb < StateMachine.GateThreshold);

                RippleOut fsmResult = StateMachine.Update(sigProb, artProb, gatesOpen, _snapshotMat);

                _currentK = (fsmResult.State == RippleState.Possible || fsmResult.State == RippleState.Ripple) ? KAtGate : KBelowGate;
                fsmResult.StrideUsed = _currentK;

                _nodeTimer.Stop();

                return new AdaptiveResult
                {
                    Ripple = fsmResult,
                    ExecutionTimeMs = _nodeTimer.Elapsed.TotalMilliseconds,
                    InferenceTimeMs = _inferenceTimer.Elapsed.TotalMilliseconds
                };
            }
        }

        private unsafe void UpdateRingBufferUnsafe(Mat mat)
        {
            float* srcPtr = (float*)mat.Data.ToPointer();
            float* ringPtr = (float*)_hRingBuffer.AddrOfPinnedObject().ToPointer();
            int cols = mat.Cols;

            for (int t = 0; t < cols; t++)
            {
                int ringIdx = (_headIndex + t) % RingBufferCapacity;
                int ringOffset = ringIdx * Channels;

                for (int c = 0; c < Channels; c++)
                {
                    ringPtr[ringOffset + c] = srcPtr[c * cols + t];
                }
            }
            _headIndex = (_headIndex + cols) % RingBufferCapacity;
        }

        private unsafe void PrepareInputUnsafe()
        {
            float* dstPtr = (float*)_hIn.AddrOfPinnedObject().ToPointer();
            float* ringPtr = (float*)_hRingBuffer.AddrOfPinnedObject().ToPointer();

            for (int t = 0; t < TimePoints; t++)
            {
                int stepsBack = (TimePoints - 1 - t) * DownsampleFactor;
                int ringIdx = _headIndex - 1 - stepsBack;

                while (ringIdx < 0) ringIdx += RingBufferCapacity;
                int ringOffset = ringIdx * Channels;

                for (int c = 0; c < Channels; c++)
                {
                    dstPtr[(t * Channels) + c] = ringPtr[ringOffset + c];
                }
            }
        }

        private unsafe void ExtractSnapshotUnsafe()
        {
            float* ringPtr = (float*)_hRingBuffer.AddrOfPinnedObject().ToPointer();
            float* dstPtr = (float*)_hSnapshotBuffer.AddrOfPinnedObject().ToPointer();

            for (int t = 0; t < TimePoints; t++)
            {
                int ringIdx = _headIndex - 1 - (TimePoints - 1 - t);
                while (ringIdx < 0) ringIdx += RingBufferCapacity;
                int ringOffset = ringIdx * Channels;

                for (int c = 0; c < Channels; c++)
                {
                    dstPtr[c * TimePoints + t] = ringPtr[ringOffset + c];
                }
            }
        }

        public void Dispose()
        {
            if (_hIn.IsAllocated) _hIn.Free();
            if (_hOut.IsAllocated) _hOut.Free();
            if (_hRingBuffer.IsAllocated) _hRingBuffer.Free();
            if (_hSnapshotBuffer.IsAllocated) _hSnapshotBuffer.Free();

            if (_inputValues != null) foreach (var v in _inputValues) v?.Dispose();
            if (_outputValues != null) foreach (var v in _outputValues) v?.Dispose();

            _runOpts?.Dispose();
            _session?.Dispose();
        }
    }
}