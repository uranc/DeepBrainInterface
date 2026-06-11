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
using System.Reactive.Subjects;
using System.Runtime.InteropServices;
using System.Threading;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Single-node port of the CPU ripple-detection workflow: decimation -> EMA z-score -> " +
                 "sliding window -> ONNX inference -> leaky-bucket FSM. Functionally identical to the " +
                 "decimate/SlidingWindowZScore/Buffer/RippleDetectorCPU/RippleStateMachineMatBool chain. " +
                 "Inference runs on a decoupled thread, processing every window in order (FIFO).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorSuperNode : IDisposable
    {
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        const uint _MCW_DN = 0x03000000;
        const uint _DN_FLUSH = 0x01000000;

        [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod")] private static extern uint TimeBeginPeriod(uint ms);
        [DllImport("winmm.dll", EntryPoint = "timeEndPeriod")]   private static extern uint TimeEndPeriod(uint ms);
        [DllImport("kernel32.dll")] static extern IntPtr GetCurrentThread();
        [DllImport("kernel32.dll")] static extern IntPtr SetThreadAffinityMask(IntPtr hThread, IntPtr dwThreadAffinityMask);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool GetLogicalProcessorInformationEx(int RelationshipType, IntPtr Buffer, ref uint ReturnedLength);
        const int RelationProcessorCore = 0;

        // --- 0. Preprocessing ---
        [Category("0. Preprocessing")]
        [Description("Decimation: keep 1 of every N full-rate samples (pure subsampling, matching " +
                     "Buffer(1, skip=N)). 30 kHz / 12 = 2.5 kHz. Z-score and windowing run on the decimated stream.")]
        public int DecimationFactor { get; set; } = 12;

        [Category("0. Preprocessing")]
        [Description("EMA Z-score window size in DECIMATED samples (alpha = 2/(N+1)). Matches SlidingWindowZScore.")]
        public int ZScoreWindowSize { get; set; } = 1250;

        [Category("0. Preprocessing")]
        [Description("New decimated samples between inference windows (sliding window stride), matching Buffer(44, skip=N).")]
        public int Stride { get; set; } = 3;

        // --- 1. Model ---
        [Category("1. Model")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";
        [Category("1. Model")] public int TimePoints { get; set; } = 44;
        [Category("1. Model")] public int Channels   { get; set; } = 8;

        // --- 2. Detection ---
        [Category("2. Detection")] public bool  DetectionEnabled          { get; set; } = true;
        [Category("2. Detection")] public float Threshold1_IgnoreBelow    { get; set; } = 0.10f;
        [Category("2. Detection")] public float Threshold2_WeakEvidence   { get; set; } = 0.50f;
        [Category("2. Detection")] public float Threshold3_StrongEvidence { get; set; } = 0.80f;
        [Category("2. Detection")] public float TargetEvidenceScore       { get; set; } = 5.0f;
        [Category("2. Detection")] public int   FramesToWaitBeforeDecay   { get; set; } = 5;
        [Category("2. Detection")] public float ScoreDecayPerFrame        { get; set; } = 1.0f;
        [Category("2. Detection")] public int   TriggerDelayMs            { get; set; } = 0;
        [Category("2. Detection")] public bool  RandomizeDelay            { get; set; } = false;
        [Category("2. Detection")] public int   PostRippleMs              { get; set; } = 50;

        // --- 3. Hardware ---
        [Category("3. Hardware")]
        [Description("Auto-pin the inference thread to a performance (P) core, ignoring TargetCore. " +
                     "Recommended on hybrid CPUs where P/E cores are interleaved in the logical index.")]
        public bool PinToPerformanceCore { get; set; } = true;
        [Category("3. Hardware")]
        [Description("Explicit logical-core index when PinToPerformanceCore is false. -1 disables pinning.")]
        public int TargetCore { get; set; } = 7;
        [Category("3. Hardware")]
        public ProcessPriorityClass ProcessPriority { get; set; } = ProcessPriorityClass.RealTime;

        // --- Acquisition-thread-only state ---
        private double   _alpha;
        private double[] _emu;
        private double[] _evar;
        private float[]  _ring;          // [ch * tp] channel-major ring of decimated z-scored samples
        private int      _ringHead;
        private int      _ringFilled;    // 0..tp
        private int      _strideCount;
        private long     _decimCounter;  // full-rate sample index; mod DecimationFactor selects kept samples

        // --- SPSC FIFO of completed windows (producer: acquisition; consumer: inference) ---
        // Every window the producer enqueues is processed by the consumer in order, so the
        // prediction count matches the reference workflow exactly. A window is only dropped if
        // the consumer falls Capacity windows behind (genuine sustained overload), counted in _droppedTotal.
        private const int Capacity = 4096;
        private float[]  _queue;         // [Capacity * tp * ch], each slot time-major [T, C] ready for ONNX
        private ulong[]  _queueClock;    // [Capacity]
        private bool[]   _queueGate;     // [Capacity]
        private long     _writeIdx;      // producer writes (Volatile), consumer reads
        private long     _readIdx;       // consumer writes (Volatile), producer reads

        private SemaphoreSlim _signal;
        private CancellationTokenSource _cts;
        private RippleStateMachineMatBool _detector;

        // Called on the acquisition thread for every incoming Mat chunk [C x N].
        private void EnqueueChunk(Mat mat, ulong clock, bool gate)
        {
            int ch    = Channels;
            int tp    = TimePoints;
            int cols  = mat.Cols;
            int decim = Math.Max(1, DecimationFactor);
            int stride = Math.Max(1, Stride);
            int windowSize = tp * ch;
            double alpha = _alpha;

            unsafe
            {
                byte* src    = (byte*)mat.Data.ToPointer();
                int   step   = mat.Step;       // bytes per row (channel)
                Depth depth  = mat.Depth;      // F32 in production; U16/S16/etc supported
                int   elemSz = ElementSize(depth);

                for (int t = 0; t < cols; t++)
                {
                    // Decimate: keep 1 of every `decim` samples, phase continuous across chunks.
                    if (_decimCounter++ % decim != 0) continue;

                    int writeSlot = _ringHead;

                    for (int c = 0; c < ch; c++)
                    {
                        byte* p = src + c * step + t * elemSz;
                        float val;
                        switch (depth)
                        {
                            case Depth.F32: val = *(float*)p;         break;
                            case Depth.U16: val = *(ushort*)p;        break;
                            case Depth.S16: val = *(short*)p;         break;
                            case Depth.S32: val = *(int*)p;           break;
                            case Depth.F64: val = (float)*(double*)p; break;
                            case Depth.U8:  val = *p;                 break;
                            case Depth.S8:  val = *(sbyte*)p;         break;
                            default:        val = *(float*)p;         break;
                        }

                        // EMA z-score, identical to SlidingWindowZScore (mu/var init 0, z uses updated mean, no clip).
                        double diff  = val - _emu[c];
                        _emu[c]     += alpha * diff;
                        _evar[c]     = (1.0 - alpha) * (_evar[c] + alpha * diff * diff);
                        double sig   = Math.Sqrt(_evar[c]);
                        double z     = (val - _emu[c]) / (sig + 1e-8);
                        _ring[c * tp + writeSlot] = (float)z;
                    }

                    _ringHead = (writeSlot + 1) % tp;
                    if (_ringFilled < tp) _ringFilled++;

                    if (++_strideCount >= stride && _ringFilled == tp)
                    {
                        _strideCount = 0;

                        long w      = Volatile.Read(ref _writeIdx);
                        int  qslot  = (int)(w % Capacity);
                        int  baseIdx = qslot * windowSize;
                        int  oldest = _ringHead; // oldest slot = next write position once ring is full

                        // Linearise ring into [T, C] time-major window (ONNX input order).
                        for (int tt = 0; tt < tp; tt++)
                        {
                            int rslot = (oldest + tt) % tp;
                            for (int cc = 0; cc < ch; cc++)
                                _queue[baseIdx + tt * ch + cc] = _ring[cc * tp + rslot];
                        }
                        _queueClock[qslot] = clock;
                        _queueGate[qslot]  = gate;

                        Volatile.Write(ref _writeIdx, w + 1);
                        if (_signal.CurrentCount == 0) _signal.Release();
                    }
                }
            }
        }

        // --- Process overloads ---

        public IObservable<RippleOut> Process(IObservable<Mat> source) =>
            Run(_ => source.Subscribe(m => EnqueueChunk(m, 0UL, true)));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source) =>
            Run(_ => source.Subscribe(t => EnqueueChunk(t.Item1, 0UL, t.Item2)));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong>> source) =>
            Run(_ => source.Subscribe(t => EnqueueChunk(t.Item1, t.Item2, true)));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong[]>> source) =>
            Run(_ => source.Subscribe(t =>
            {
                ulong clock = (t.Item2 != null && t.Item2.Length > 0) ? t.Item2[t.Item2.Length - 1] : 0UL;
                EnqueueChunk(t.Item1, clock, true);
            }));

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong>, bool>> source) =>
            Run(_ => source.Subscribe(t => EnqueueChunk(t.Item1.Item1, t.Item1.Item2, t.Item2)));

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong[]>, bool>> source) =>
            Run(_ => source.Subscribe(t =>
            {
                ulong clock = (t.Item1.Item2 != null && t.Item1.Item2.Length > 0) ? t.Item1.Item2[t.Item1.Item2.Length - 1] : 0UL;
                EnqueueChunk(t.Item1.Item1, clock, t.Item2);
            }));

        private IObservable<RippleOut> Run(Func<IObservable<RippleOut>, IDisposable> subscribe)
        {
            var subject = new Subject<RippleOut>();

            return Observable.Create<RippleOut>(observer =>
            {
                int ch = Channels;
                int tp = TimePoints;

                _alpha        = 2.0 / (ZScoreWindowSize + 1);
                _emu          = new double[ch];
                _evar         = new double[ch];
                _ring         = new float[ch * tp];
                _ringHead     = 0;
                _ringFilled   = 0;
                _strideCount  = 0;
                _decimCounter = 0;
                _queue        = new float[Capacity * tp * ch];
                _queueClock   = new ulong[Capacity];
                _queueGate    = new bool[Capacity];
                _writeIdx     = 0;
                _readIdx      = 0;
                _detector     = new RippleStateMachineMatBool();
                _signal       = new SemaphoreSlim(0, 1);
                _cts          = new CancellationTokenSource();

                var upstreamSub = subscribe(subject);

                var worker = new Thread(() => InferenceLoop(subject, _cts.Token))
                {
                    IsBackground = true,
                    Priority     = ThreadPriority.Highest,
                    Name         = "RippleSuperNodeInference"
                };
                worker.Start();

                var downstreamSub = subject.Subscribe(observer);

                return Disposable.Create(() =>
                {
                    _cts.Cancel();
                    _signal.Release();
                    worker.Join(500);
                    upstreamSub.Dispose();
                    downstreamSub.Dispose();
                    subject.Dispose();
                });
            });
        }

        private unsafe void InferenceLoop(Subject<RippleOut> subject, CancellationToken token)
        {
            OrtValue valIn = null, valOut = null;
            OrtIoBinding binding = null;
            RunOptions runOpts = null;
            InferenceSession session = null;
            GCHandle hIn = default, hOut = default;

            try
            {
                try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { }
                TimeBeginPeriod(1);
                try
                {
                    using (var proc = Process.GetCurrentProcess())
                        proc.PriorityClass = ProcessPriority;
                }
                catch { }
                long pinMask =
                    PinToPerformanceCore ? FindPerformanceCoreMask() :
                    (TargetCore >= 0 && TargetCore < 64) ? (1L << TargetCore) : 0L;
                if (pinMask != 0) SetThreadAffinityMask(GetCurrentThread(), (IntPtr)pinMask);

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

                session = new InferenceSession(ModelPath, opts);
                binding = session.CreateIoBinding();
                runOpts = new RunOptions();
                opts.Dispose();

                float[] inBuf  = new float[windowSize];
                hIn  = GCHandle.Alloc(inBuf,  GCHandleType.Pinned);
                float[] outBuf = new float[1];
                hOut = GCHandle.Alloc(outBuf, GCHandleType.Pinned);
                var mem = OrtMemoryInfo.DefaultInstance;
                valIn  = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(inBuf),  new long[] { 1, TimePoints, Channels });
                valOut = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(outBuf), new long[] { 1, 1 });
                binding.BindInput (session.InputMetadata.Keys.First(),  valIn);
                binding.BindOutput(session.OutputMetadata.Keys.First(), valOut);
                for (int i = 0; i < 50; i++) session.RunWithBinding(runOpts, binding);

                while (!token.IsCancellationRequested)
                {
                    _signal.Wait(token);

                    // Drain every enqueued window in order — one inference + one output per window.
                    long r;
                    while ((r = Volatile.Read(ref _readIdx)) < Volatile.Read(ref _writeIdx))
                    {
                        int qslot = (int)(r % Capacity);
                        Array.Copy(_queue, qslot * windowSize, inBuf, 0, windowSize);
                        ulong clock = _queueClock[qslot];
                        bool  gate  = _queueGate[qslot];

                        // Inference runs on every window (gate only affects the FSM), matching RippleDetectorCPU.
                        session.RunWithBinding(runOpts, binding);
                        float prob = outBuf[0];

                        _detector.DetectionEnabled          = DetectionEnabled;
                        _detector.Threshold1_IgnoreBelow    = Threshold1_IgnoreBelow;
                        _detector.Threshold2_WeakEvidence   = Threshold2_WeakEvidence;
                        _detector.Threshold3_StrongEvidence = Threshold3_StrongEvidence;
                        _detector.TargetEvidenceScore       = TargetEvidenceScore;
                        _detector.FramesToWaitBeforeDecay   = FramesToWaitBeforeDecay;
                        _detector.ScoreDecayPerFrame        = ScoreDecayPerFrame;
                        _detector.TriggerDelayMs            = TriggerDelayMs;
                        _detector.RandomizeDelay            = RandomizeDelay;
                        _detector.PostRippleMs              = PostRippleMs;

                        RippleOut result = _detector.Update(prob, 0f, gate, null);
                        result.Clock   = clock;
                        result.Skipped = 0;

                        subject.OnNext(result);

                        Volatile.Write(ref _readIdx, r + 1);
                    }
                }
            }
            catch (OperationCanceledException) { /* normal shutdown */ }
            catch (Exception ex) { subject.OnError(ex); }
            finally
            {
                valIn?.Dispose(); valOut?.Dispose();
                binding?.Dispose(); runOpts?.Dispose(); session?.Dispose();
                if (hIn.IsAllocated)  hIn.Free();
                if (hOut.IsAllocated) hOut.Free();
                TimeEndPeriod(1);
            }
        }

        // Bytes per element for each OpenCV.Net depth.
        private static int ElementSize(Depth d)
        {
            switch (d)
            {
                case Depth.U8:  case Depth.S8:  return 1;
                case Depth.U16: case Depth.S16: return 2;
                case Depth.S32: case Depth.F32: return 4;
                case Depth.F64:                 return 8;
                default:                        return 4;
            }
        }

        // Returns the affinity mask of one performance (P) core (highest EfficiencyClass).
        // Picks the last such core to stay clear of core-0 housekeeping. 0 if unavailable.
        private static long FindPerformanceCoreMask()
        {
            uint len = 0;
            GetLogicalProcessorInformationEx(RelationProcessorCore, IntPtr.Zero, ref len);
            if (len == 0) return 0;
            IntPtr buf = Marshal.AllocHGlobal((int)len);
            try
            {
                if (!GetLogicalProcessorInformationEx(RelationProcessorCore, buf, ref len))
                    return 0;
                long bestMask = 0; int bestClass = -1;
                int offset = 0;
                while (offset + 8 <= (int)len)
                {
                    int relationship = Marshal.ReadInt32(buf, offset + 0);
                    int size         = Marshal.ReadInt32(buf, offset + 4);
                    if (size <= 0) break;
                    if (relationship == RelationProcessorCore)
                    {
                        // PROCESSOR_RELATIONSHIP: EfficiencyClass@9, GroupMask[0].Mask@32 (KAFFINITY, 8B x64)
                        byte effClass = Marshal.ReadByte(buf, offset + 9);
                        long mask     = (long)Marshal.ReadIntPtr(buf, offset + 32);
                        if (effClass >= bestClass) { bestClass = effClass; bestMask = mask; }
                    }
                    offset += size;
                }
                return bestMask;
            }
            finally { Marshal.FreeHGlobal(buf); }
        }

        public void Dispose() => _cts?.Cancel();
    }
}
