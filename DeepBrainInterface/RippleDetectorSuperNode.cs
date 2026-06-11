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
    [Description("Streaming SuperNode: EMA Z-score + sliding window + ONNX inference in one node. " +
                 "Accepts raw F32 µV Mat chunks [C×N] — no external Buffer or ZScore nodes needed. " +
                 "Acquisition thread maintains ring buffer; inference thread always picks up newest window.")]
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
        [Description("Decimation: keep 1 of every N full-rate samples (pure subsampling, " +
                     "matching Buffer(1, skip=N) in the reference workflow). 30 kHz / 12 = 2.5 kHz. " +
                     "Z-score and windowing run on the DECIMATED stream.")]
        public int DecimationFactor { get; set; } = 12;

        [Category("0. Preprocessing")]
        [Description("EMA Z-score window size in DECIMATED samples (alpha = 2/(N+1)).")]
        public int ZScoreWindowSize { get; set; } = 1250;

        [Category("0. Preprocessing")]
        [Description("Symmetric clip threshold for Z-score output (±value).")]
        public float ZScoreClip { get; set; } = 8f;

        [Category("0. Preprocessing")]
        [Description("Number of new samples between inference triggers (sliding window stride).")]
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
                     "Recommended on hybrid CPUs (Arrow Lake etc.) where P/E cores are interleaved in the logical index.")]
        public bool PinToPerformanceCore { get; set; } = true;
        [Category("3. Hardware")]
        [Description("Explicit logical-core index for the inference thread when PinToPerformanceCore is false. -1 disables pinning.")]
        public int TargetCore { get; set; } = 7;
        [Category("3. Hardware")]
        public ProcessPriorityClass ProcessPriority { get; set; } = ProcessPriorityClass.RealTime;

        // --- Acquisition-thread-only state ---
        // Ring buffer: _ring[c * TimePoints + slot] = channel c at circular slot.
        // _ringHead is the next write slot (oldest data lives there after wrap).
        private float[]  _ring;
        private int      _ringHead;
        private int      _ringFilled;  // 0..TimePoints; capped once full
        private int      _strideCount;
        private long     _decimCounter;  // full-rate sample index, mod DecimationFactor selects kept samples
        private double   _alpha;
        private double[] _emu;
        private double[] _evar;

        // Single reusable [T*C] snapshot buffer guarded by a seqlock.
        // The writer bumps _snapSeq to odd before touching the buffer and back
        // to even when done; a reader that sees an odd count, or a count that
        // changed across its copy, retries. This is allocation-free AND immune
        // to the writer lapping the reader during a multi-ms stall (newest-wins:
        // a retry just grabs even fresher data).
        private float[] _snap;     // [T, C] order, ready to feed ONNX
        private int     _snapSeq;  // even = stable, odd = write in progress

        // Shared (written by acq, read by inference via volatile / Interlocked)
        private ulong         _latestClock;
        private volatile bool _gateOpen = true;
        private int           _frameCounter;
        private SemaphoreSlim _signal;
        private CancellationTokenSource _cts;
        private RippleStateMachineMatBool _detector;
        private bool _lastTTL;

        // Called on the acquisition thread for every incoming Mat chunk [C×N].
        private void EnqueueChunk(Mat mat, ulong clock, bool gate)
        {
            int ch   = Channels;
            int tp   = TimePoints;
            int cols = mat.Cols;
            int decim = Math.Max(1, DecimationFactor);
            double alpha = _alpha;
            float clip   = ZScoreClip;

            unsafe
            {
                byte* src   = (byte*)mat.Data.ToPointer();
                int   step  = mat.Step;        // bytes per row (channel)
                Depth depth = mat.Depth;       // input is U16 (raw RHD) in the reference wiring; may be S16/F32
                int   elemSz = ElementSize(depth);

                for (int t = 0; t < cols; t++)
                {
                    // Pure subsample: keep 1 of every `decim` full-rate samples.
                    // Counter is continuous across chunks, matching Bonsai Buffer's
                    // carried skip counter — z-score/window then see the 2.5 kHz stream.
                    if (_decimCounter++ % decim != 0) continue;

                    int writeSlot = _ringHead;

                    for (int c = 0; c < ch; c++)
                    {
                        // Read the element in its native depth. Any positive affine
                        // (U16 raw vs µV-scaled F32) is normalised away by the z-score
                        // below, so no µV ConvertScale is needed upstream.
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
                        double diff  = val - _emu[c];
                        _emu[c]     += alpha * diff;
                        _evar[c]     = (1.0 - alpha) * (_evar[c] + alpha * diff * diff);
                        double sig   = Math.Sqrt(_evar[c]);
                        // Match SlidingWindowZScore: z uses updated mean
                        double z     = (val - _emu[c]) / (sig + 1e-8);
                        if      (z >  clip) z =  clip;
                        else if (z < -clip) z = -clip;
                        _ring[c * tp + writeSlot] = (float)z;
                    }

                    _ringHead = (writeSlot + 1) % tp;
                    if (_ringFilled < tp) _ringFilled++;

                    if (++_strideCount >= Stride && _ringFilled == tp)
                    {
                        _strideCount = 0;

                        // Seqlock write: odd while filling, even when published.
                        Volatile.Write(ref _snapSeq, _snapSeq + 1); // -> odd

                        float[] snap   = _snap;
                        int     oldest = _ringHead; // oldest slot = next write slot
                        for (int tt = 0; tt < tp; tt++)
                        {
                            int slot = (oldest + tt) % tp;
                            for (int cc = 0; cc < ch; cc++)
                                snap[tt * ch + cc] = _ring[cc * tp + slot];
                        }
                        _latestClock = clock;
                        _gateOpen    = gate;

                        Volatile.Write(ref _snapSeq, _snapSeq + 1); // -> even

                        Interlocked.Increment(ref _frameCounter);
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

                // Reset all acquisition-thread state
                _alpha       = 2.0 / (ZScoreWindowSize + 1);
                _emu         = new double[ch];
                _evar        = new double[ch];
                _ring        = new float[ch * tp];
                _ringHead     = 0;
                _ringFilled   = 0;
                _strideCount  = 0;
                _decimCounter = 0;
                _snap         = new float[tp * ch];
                _snapSeq     = 0;
                _latestClock = 0;
                _gateOpen    = true;
                _lastTTL     = false;
                _frameCounter = 0;
                _detector    = new RippleStateMachineMatBool();
                _signal      = new SemaphoreSlim(0, 1);
                _cts         = new CancellationTokenSource();

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

                int lastFrame = 0;

                while (!token.IsCancellationRequested)
                {
                    _signal.Wait(token);

                    int currentFrame = Volatile.Read(ref _frameCounter);
                    if (currentFrame == lastFrame) continue;

                    int skippedFrames = currentFrame - lastFrame - 1;
                    lastFrame = currentFrame;

                    // Seqlock read: copy the snapshot, retry if the writer touched
                    // it mid-copy. newest-wins, so a retry just grabs fresher data.
                    // The copy is unconditional (1.4 KB, ~µs) to keep the (window,
                    // clock, gate) triple consistent even when the gate is closed.
                    ulong clock;
                    bool  gate;
                    int   spins = 0;
                    while (true)
                    {
                        int seqBefore = Volatile.Read(ref _snapSeq);
                        if ((seqBefore & 1) == 0)            // writer not mid-write
                        {
                            Array.Copy(_snap, inBuf, windowSize);  // [T, C], ready for ONNX
                            clock = _latestClock;
                            gate  = _gateOpen;
                            if (Volatile.Read(ref _snapSeq) == seqBefore) break;
                        }
                        if (++spins > 1000) { clock = _latestClock; gate = _gateOpen; break; }
                    }

                    float prob = 0f;
                    if (gate && !_lastTTL)
                    {
                        session.RunWithBinding(runOpts, binding);
                        prob = outBuf[0];
                    }

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
                    result.Skipped = skippedFrames + (_lastTTL ? 1 : 0);
                    _lastTTL = result.TTL;

                    subject.OnNext(result);
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

        // Walks the OS processor topology and returns the affinity mask of a single
        // performance (P) core — the one with the highest EfficiencyClass (0 = most
        // efficient/E-core, higher = higher-performance/P-core). Picks the LAST such
        // core (highest logical index) to stay clear of core-0 OS/driver housekeeping.
        // On hybrid parts (Arrow Lake etc.) P and E cores are interleaved in the
        // logical index, so a hardcoded index is unreliable — this resolves a real
        // P-core regardless of enumeration. Returns 0 if topology can't be read.
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
                        // PROCESSOR_RELATIONSHIP: Flags@8, EfficiencyClass@9,
                        // Reserved[20]@10, GroupCount@30, GroupMask[0].Mask@32 (KAFFINITY, 8B on x64)
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
