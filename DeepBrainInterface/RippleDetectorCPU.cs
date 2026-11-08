using Bonsai;
using DeepBrainInterface.Core;
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
        public int OutputRingDepth { get; set; } = 0;   // 0 = allocate a fresh Mat per inference (1.1.65 behaviour)

        private readonly object _inferenceLock = new object();
        private Mat[] _outRing;
        private int _outIdx;
        private bool _hotThreadBoosted;
        private bool _hotThreadFpuSet;
        private OrtRunner _runner;
        private int _inputStride, _expectedBytes, _expectedContiguousStep, _outCols;
        private readonly List<Mat> _inputCache = new List<Mat>(2);

        private void InitializeSession()
        {
            if (_runner != null) return;
            _inputStride = TimePoints * Channels;
            _expectedBytes = _inputStride * sizeof(float);
            _expectedContiguousStep = Channels * sizeof(float);
            try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { }
            if (ManagePriority)
            {
                try { using (var process = System.Diagnostics.Process.GetCurrentProcess()) process.PriorityClass = System.Diagnostics.ProcessPriorityClass.RealTime; } catch { }
            }
            _runner = new OrtRunner(ModelPath, BatchSize, TimePoints, Channels);   // bound pinned buffers, 50 warm-up runs
            _outCols = 1;
            _outRing = null; _outIdx = 0;
            if (OutputRingDepth > 0)
            {
                int ringDepth = Math.Max(2, OutputRingDepth);
                _outRing = new Mat[ringDepth];
                for (int i = 0; i < ringDepth; i++) _outRing[i] = new Mat(BatchSize, _outCols, Depth.F32, 1);
            }
        }

        private Mat RunInference(IList<Mat> mats)
        {
            lock (_inferenceLock)
            {
                if (_runner == null) InitializeSession();
                if (!_hotThreadFpuSet) { try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { } _hotThreadFpuSet = true; }
                if (ManagePriority && !_hotThreadBoosted) { try { System.Threading.Thread.CurrentThread.Priority = System.Threading.ThreadPriority.Highest; } catch { } _hotThreadBoosted = true; }
                unsafe
                {
                    fixed (float* dstBase = _runner.Input)
                    {
                        for (int i = 0; i < mats.Count; i++)
                        {
                            Mat inputMat = mats[i];
                            byte* srcBase = (byte*)inputMat.Data.ToPointer();
                            float* dst = dstBase + (i * _inputStride);
                            if (inputMat.Rows == TimePoints && inputMat.Cols == Channels && inputMat.Step == _expectedContiguousStep)
                            {
                                Buffer.MemoryCopy(srcBase, dst, _expectedBytes, _expectedBytes);
                            }
                            else if (inputMat.Rows == Channels && inputMat.Cols == TimePoints)
                            {
                                int step = inputMat.Step;
                                for (int c = 0; c < Channels; c++)
                                {
                                    float* srcRow = (float*)(srcBase + (c * step));
                                    for (int t = 0; t < TimePoints; t++) dst[(t * Channels) + c] = srcRow[t];
                                }
                            }
                            else
                            {
                                int step = inputMat.Step, cols = inputMat.Cols, rowBytes = cols * sizeof(float);
                                for (int r = 0; r < inputMat.Rows; r++) Buffer.MemoryCopy(srcBase + (r * step), (byte*)(dst + (r * cols)), rowBytes, rowBytes);
                            }
                        }
                    }
                }
                _runner.Run();
                Mat resultMat;
                if (_outRing == null) resultMat = new Mat(BatchSize, _outCols, Depth.F32, 1);   // 1.1.65 path
                else { resultMat = _outRing[_outIdx]; _outIdx = (_outIdx + 1 == _outRing.Length) ? 0 : _outIdx + 1; }
                unsafe
                {
                    float* o = (float*)resultMat.Data.ToPointer();
                    for (int b = 0; b < BatchSize; b++) o[b] = _runner.Output[b];
                }
                return resultMat;
            }
        }

        // Kept for callers that want an explicit warm-up; NOT called at subscribe any more (1.1.73): creating the
        // session on the subscribing thread left the acquisition thread without FTZ/DAZ (ORT sets it on the
        // initialising thread only) and the 11-03 01:06 run fell to ~1/4 real time.
        private void EnsureInitialized()
        {
            lock (_inferenceLock)
            {
                if (_runner == null) InitializeSession();
            }
        }

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Defer(() =>
            {
                // Session is created lazily on the first data frame, on the acquisition thread (as in 1.1.65):
                // ORT and _controlfp_s then apply FTZ/DAZ to the thread that actually runs inference.
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
                // Session is created lazily on the first data frame, on the acquisition thread (as in 1.1.65):
                // ORT and _controlfp_s then apply FTZ/DAZ to the thread that actually runs inference.
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
                // Session is created lazily on the first data frame, on the acquisition thread (as in 1.1.65):
                // ORT and _controlfp_s then apply FTZ/DAZ to the thread that actually runs inference.
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
                // Session is created lazily on the first data frame, on the acquisition thread (as in 1.1.65):
                // ORT and _controlfp_s then apply FTZ/DAZ to the thread that actually runs inference.
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
                // Session is created lazily on the first data frame, on the acquisition thread (as in 1.1.65):
                // ORT and _controlfp_s then apply FTZ/DAZ to the thread that actually runs inference.
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
            _runner?.Dispose(); _runner = null;
            if (_outRing != null) { for (int i = 0; i < _outRing.Length; i++) _outRing[i]?.Dispose(); _outRing = null; }
            _hotThreadBoosted = false; _hotThreadFpuSet = false;
        }

        public void Dispose()
        {
            lock (_inferenceLock) { ReleaseAllocations(); }
        }
    }
}