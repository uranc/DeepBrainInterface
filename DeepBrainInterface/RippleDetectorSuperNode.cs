using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace DeepBrainInterface
{
    // Drop-in replacement for RippleDetectorGPU + RippleStateMachineMatBool.
    //
    // What's the same as the existing workflow:
    //   - Accepts the same windowed z-scored [Channels x TimePoints] Mat (output of Buffer(44,3))
    //   - Accepts the same BNO gate (bool from BitwiseNot -> StartWithBool(false))
    //   - Outputs identical RippleOut (same struct, same FSM via RippleStateMachineMatBool.Update)
    //
    // What's new:
    //   - Acquisition thread writes windows into a small lock-free ring instead of blocking on ONNX
    //   - Background thread (pinned, RealTime) runs ONNX + FSM; skips ONNX during TTL refractory
    //   - Accepts optional ulong clock per window so RippleOut.Clock tracks which frame triggered detection
    //
    // Wiring (replaces GPU node + FSM node):
    //   Buffer(44,3)
    //     -> [optional] Zip with Rhd2164.Clock       => Tuple<Mat, ulong>
    //     -> [optional] WithLatestFrom(BNO gate)      => Tuple<Tuple<Mat, ulong>, bool>
    //     -> RippleDetectorSuperNode
    //     -> TTL -> DigitalOutput

    [Combinator]
    [Description("Ring-buffered drop-in for RippleDetectorGPU + RippleStateMachineMatBool. " +
                 "Acquisition thread writes windowed z-scored Mats into a lock-free ring; " +
                 "a pinned RealTime background thread runs zero-alloc ONNX + FSM, " +
                 "skipping ONNX during the TTL refractory. Output is identical RippleOut.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorSuperNode : IDisposable
    {
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        const uint _MCW_DN = 0x03000000;
        const uint _DN_FLUSH = 0x01000000;

        [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod")] private static extern uint TimeBeginPeriod(uint ms);
        [DllImport("winmm.dll", EntryPoint = "timeEndPeriod")] private static extern uint TimeEndPeriod(uint ms);
        [DllImport("kernel32.dll")] static extern IntPtr GetCurrentThread();
        [DllImport("kernel32.dll")] static extern IntPtr SetThreadAffinityMask(IntPtr hThread, IntPtr dwThreadAffinityMask);

        // --- 1. Model (same as RippleDetectorGPU) ---
        [Category("1. Model")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";
        [Category("1. Model")] public int TimePoints { get; set; } = 44;
        [Category("1. Model")] public int Channels { get; set; } = 8;

        // --- 2. FSM (same settings as RippleStateMachineMatBool) ---
        [Category("2. FSM")]
        [Description("State machine — identical to RippleStateMachineMatBool. Expand to configure.")]
        [DesignerSerializationVisibility(DesignerSerializationVisibility.Content)]
        public RippleStateMachineMatBool FSM { get; set; } = new RippleStateMachineMatBool();

        // --- 3. Hardware Tuning (same as RippleDetectorGPU) ---
        [Category("3. Hardware Tuning")]
        [Description("Pin the inference thread to this core. -1 disables.")]
        public int TargetCore { get; set; } = 7;
        [Category("3. Hardware Tuning")]
        [Description("Busy-spin for lowest latency (pegs TargetCore). False = SpinOnce (yields CPU).")]
        public bool BusySpin { get; set; } = true;
        [Category("3. Hardware Tuning")]
        public ProcessPriorityClass ProcessPriority { get; set; } = ProcessPriorityClass.RealTime;

        // Ring of pre-allocated inference windows.
        // Power of 2. 32 slots @ 833 Hz = ~38 ms of buffering — enough for a GC pause.
        private const int RingSlots = 32;
        private float[][] _windowRing;   // pre-allocated [slot][TimePoints * Channels]
        private ulong[]   _clockRing;    // hardware clock per slot
        private bool[]    _gateRing;     // BNO gate per slot
        private volatile int _writeHead;
        private volatile int _readHead;

        private CancellationTokenSource _cts;
        private bool _lastTTL;           // used to detect refractory on background thread

        // ----- Process overloads — same input signatures as GPU + FSM -----

        // Same as RippleDetectorGPU: windowed z-scored [Channels x TimePoints] Mat only.
        public IObservable<RippleOut> Process(IObservable<Mat> source) =>
            Run(observer => source.Subscribe(m => Enqueue(m, 0UL, true), observer.OnError));

        // With BNO gate: mirrors WithLatestFrom(gate) -> RippleStateMachineMatBool wiring.
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source) =>
            Run(observer => source.Subscribe(t => Enqueue(t.Item1, 0UL, t.Item2), observer.OnError));

        // With hardware clock (Zip data with Rhd2164.Clock) — no gate.
        // Scalar ulong: single clock per window.
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong>> source) =>
            Run(observer => source.Subscribe(t => Enqueue(t.Item1, t.Item2, true), observer.OnError));

        // With clock array (Rhd2164.Clock is ulong[]) — no gate.
        // Extract the most recent (last) clock from the array.
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong[]>> source) =>
            Run(observer => source.Subscribe(t =>
            {
                ulong clock = (t.Item2 != null && t.Item2.Length > 0) ? t.Item2[t.Item2.Length - 1] : 0UL;
                Enqueue(t.Item1, clock, true);
            }, observer.OnError));

        // Full: Zip(data, clock) -> WithLatestFrom(BNO gate).
        // Scalar ulong.
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong>, bool>> source) =>
            Run(observer => source.Subscribe(t => Enqueue(t.Item1.Item1, t.Item1.Item2, t.Item2), observer.OnError));

        // Full with clock array: Zip(data, clock[]) -> WithLatestFrom(gate).
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong[]>, bool>> source) =>
            Run(observer => source.Subscribe(t =>
            {
                ulong clock = (t.Item1.Item2 != null && t.Item1.Item2.Length > 0) ? t.Item1.Item2[t.Item1.Item2.Length - 1] : 0UL;
                Enqueue(t.Item1.Item1, clock, t.Item2);
            }, observer.OnError));

        private IObservable<RippleOut> Run(Func<IObserver<RippleOut>, IDisposable> subscribe)
        {
            return Observable.Create<RippleOut>(observer =>
            {
                int windowSize = TimePoints * Channels;
                _windowRing = new float[RingSlots][];
                for (int i = 0; i < RingSlots; i++) _windowRing[i] = new float[windowSize];
                _clockRing = new ulong[RingSlots];
                _gateRing  = new bool[RingSlots];
                _writeHead = 0; _readHead = 0; _lastTTL = false;

                _cts = new CancellationTokenSource();
                var sub = subscribe(observer);

                var worker = new Thread(() => RunInference(observer, _cts.Token))
                {
                    IsBackground = true,
                    Priority = ThreadPriority.Highest,
                    Name = "RippleSuperNodeInference"
                };
                worker.Start();

                return Disposable.Create(() =>
                {
                    _cts.Cancel();
                    worker.Join(500);
                    sub.Dispose();
                });
            });
        }

        // ----- Writer: runs on the acquisition thread -----
        // Copies the incoming windowed Mat into the next ring slot.
        // If the ring is full (background thread is stalled) the oldest slot is overwritten.
        private void Enqueue(Mat mat, ulong clock, bool gate)
        {
            int slot = _writeHead & (RingSlots - 1);
            float[] dst = _windowRing[slot];

            unsafe
            {
                // Input is [Channels x TimePoints] channel-major (same as RippleDetectorGPU).
                // Model expects time-major [1, TimePoints, Channels]: transpose on the fly.
                byte* src  = (byte*)mat.Data.ToPointer();
                int   step = mat.Step;
                int   C    = Channels, T = TimePoints;

                fixed (float* pDst = dst)
                    for (int c = 0; c < C; c++)
                    {
                        float* row = (float*)(src + c * step);
                        for (int t = 0; t < T; t++)
                            pDst[t * C + c] = row[t];
                    }
            }

            _clockRing[slot] = clock;
            _gateRing[slot]  = gate;
            _writeHead++;   // volatile int: increment publishes the slot
        }

        // ----- Background thread: spin-poll ring -> ONNX -> FSM -----
        private unsafe void RunInference(IObserver<RippleOut> observer, CancellationToken token)
        {
            try { uint cc = 0; _controlfp_s(ref cc, _DN_FLUSH, _MCW_DN); } catch { }
            TimeBeginPeriod(1);
            try
            {
                using (var proc = System.Diagnostics.Process.GetCurrentProcess())
                    proc.PriorityClass = ProcessPriority;
            }
            catch { }
            if (TargetCore >= 0 && TargetCore < 64)
                SetThreadAffinityMask(GetCurrentThread(), (IntPtr)(1L << TargetCore));

            int windowSize = TimePoints * Channels;

            var opts = new SessionOptions
            {
                IntraOpNumThreads = 1, InterOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                EnableCpuMemArena = true, EnableMemoryPattern = true
            };
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            opts.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");

            var session = new InferenceSession(ModelPath, opts);
            var binding = session.CreateIoBinding();
            var runOpts = new RunOptions();
            opts.Dispose();

            float[] inBuf = new float[windowSize];
            var hIn = GCHandle.Alloc(inBuf, GCHandleType.Pinned);
            float[] outBuf = new float[1];
            var hOut = GCHandle.Alloc(outBuf, GCHandleType.Pinned);
            var mem = OrtMemoryInfo.DefaultInstance;
            var valIn  = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(inBuf),  new long[] { 1, TimePoints, Channels });
            var valOut = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(outBuf), new long[] { 1, 1 });
            binding.BindInput (session.InputMetadata.Keys.First(),  valIn);
            binding.BindOutput(session.OutputMetadata.Keys.First(), valOut);
            for (int i = 0; i < 50; i++) session.RunWithBinding(runOpts, binding);

            var spinner = new SpinWait();

            try
            {
                while (!token.IsCancellationRequested)
                {
                    // Wait for a new window.
                    if (_readHead == _writeHead)
                    {
                        if (BusySpin) Thread.SpinWait(64); else spinner.SpinOnce();
                        continue;
                    }

                    int   slot  = _readHead & (RingSlots - 1);
                    ulong clock = _clockRing[slot];
                    bool  gate  = _gateRing[slot];
                    _readHead++;

                    float prob = 0f;

                    // Skip ONNX during TTL refractory — pass prob=0 so the FSM advances its hold timer.
                    // _lastTTL=true means either we just fired or we're still in the hold window.
                    if (!_lastTTL)
                    {
                        fixed (float* pSrc = _windowRing[slot], pDst = inBuf)
                            Buffer.MemoryCopy(pSrc, pDst, windowSize * sizeof(float), windowSize * sizeof(float));

                        session.RunWithBinding(runOpts, binding);
                        prob = outBuf[0];
                    }

                    // Identical call to the standalone FSM node in the current workflow.
                    RippleOut result  = FSM.Update(prob, 0f, gate, null);
                    result.Clock      = clock;
                    result.Skipped    = _lastTTL ? 1 : 0;  // 1 = ONNX was skipped (refractory); 0 = ran normally

                    _lastTTL = result.TTL;
                    observer.OnNext(result);
                }
            }
            finally
            {
                valIn.Dispose(); valOut.Dispose(); binding.Dispose(); runOpts.Dispose(); session.Dispose();
                if (hIn.IsAllocated) hIn.Free();
                if (hOut.IsAllocated) hOut.Free();
                TimeEndPeriod(1);
            }
        }

        public void Dispose() => _cts?.Cancel();
    }
}
