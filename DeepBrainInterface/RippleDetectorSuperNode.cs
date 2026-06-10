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
    [Description("Decoupled inference + detection. Acquisition thread stores the latest windowed Mat; " +
                 "a background thread always picks up the NEWEST frame (stale frames during hiccups are skipped). " +
                 "ONNX is skipped during refractory. Output: RippleOut with Clock, Skipped, config snapshot.")]
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
        [Description("Pin the inference thread to this core. -1 disables.")]
        public int TargetCore { get; set; } = 7;
        [Category("3. Hardware")]
        public ProcessPriorityClass ProcessPriority { get; set; } = ProcessPriorityClass.RealTime;

        // Writer (acquisition thread) stores the latest frame and bumps a counter.
        // Reader (inference thread) checks the counter — if it moved, process the latest Mat and
        // record where it left off. Frames that arrive during inference are silently overwritten;
        // the reader always picks up the newest one. Skipped = counter delta - 1.
        private volatile Mat _latestMat;
        private ulong        _latestClock;          // written before Release(), read after Wait()
        private volatile bool _gateOpen = true;
        private int          _frameCounter;         // Interlocked — incremented by writer
        private SemaphoreSlim _signal;
        private CancellationTokenSource _cts;
        private readonly RippleStateMachineMatBool _detector = new RippleStateMachineMatBool();
        private bool _lastTTL;

        private void Enqueue(Mat mat, ulong clock, bool gate)
        {
            _latestMat   = mat;
            _latestClock = clock;
            _gateOpen    = gate;
            Interlocked.Increment(ref _frameCounter);   // publish new frame count
            if (_signal.CurrentCount == 0) _signal.Release(); // wake reader if idle
        }

        // --- Process overloads ---

        public IObservable<RippleOut> Process(IObservable<Mat> source) =>
            Run(sub => sub.Subscribe(m => Enqueue(m, 0UL, true)));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source) =>
            Run(sub => source.Subscribe(t => Enqueue(t.Item1, 0UL, t.Item2)));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong>> source) =>
            Run(sub => source.Subscribe(t => Enqueue(t.Item1, t.Item2, true)));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong[]>> source) =>
            Run(sub => source.Subscribe(t =>
            {
                ulong clock = (t.Item2 != null && t.Item2.Length > 0) ? t.Item2[t.Item2.Length - 1] : 0UL;
                Enqueue(t.Item1, clock, true);
            }));

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong>, bool>> source) =>
            Run(sub => source.Subscribe(t => Enqueue(t.Item1.Item1, t.Item1.Item2, t.Item2)));

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong[]>, bool>> source) =>
            Run(sub => source.Subscribe(t =>
            {
                ulong clock = (t.Item1.Item2 != null && t.Item1.Item2.Length > 0) ? t.Item1.Item2[t.Item1.Item2.Length - 1] : 0UL;
                Enqueue(t.Item1.Item1, clock, t.Item2);
            }));

        private IObservable<RippleOut> Run(Func<IObservable<RippleOut>, IDisposable> subscribe)
        {
            // Subject is the thread-safe bridge: background thread calls OnNext,
            // Bonsai subscribers receive on whatever thread Subject dispatches to (Rx default: caller thread).
            var subject = new Subject<RippleOut>();

            return Observable.Create<RippleOut>(observer =>
            {
                _latestMat   = null;
                _latestClock = 0;
                _gateOpen      = true;
                _lastTTL       = false;
                _frameCounter  = 0;
                _signal        = new SemaphoreSlim(0, 1);
                _cts           = new CancellationTokenSource();

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
                    _signal.Release(); // unblock Wait if sitting idle
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

                session = new InferenceSession(ModelPath, opts);
                binding = session.CreateIoBinding();
                runOpts = new RunOptions();
                opts.Dispose();

                float[] inBuf  = new float[windowSize];
                hIn  = GCHandle.Alloc(inBuf,  GCHandleType.Pinned);
                float[] outBuf = new float[1];
                hOut = GCHandle.Alloc(outBuf, GCHandleType.Pinned);
                var mem  = OrtMemoryInfo.DefaultInstance;
                valIn  = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(inBuf),  new long[] { 1, TimePoints, Channels });
                valOut = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(outBuf), new long[] { 1, 1 });
                binding.BindInput (session.InputMetadata.Keys.First(),  valIn);
                binding.BindOutput(session.OutputMetadata.Keys.First(), valOut);
                for (int i = 0; i < 50; i++) session.RunWithBinding(runOpts, binding);

                int lastFrame = 0;

                while (!token.IsCancellationRequested)
                {
                    // Block until writer signals a new frame — zero CPU while idle.
                    _signal.Wait(token);

                    // Read the current counter. If it hasn't moved (spurious wake), skip.
                    int currentFrame = Volatile.Read(ref _frameCounter);
                    if (currentFrame == lastFrame) continue;

                    int skippedFrames = currentFrame - lastFrame - 1; // frames overwritten during last inference
                    lastFrame = currentFrame;

                    // Always grab the LATEST pointer — frames that arrived during inference are discarded.
                    Mat mat = _latestMat;
                    if (mat == null) continue;

                    ulong clock = _latestClock;
                    bool  gate  = _gateOpen;

                    float prob = 0f;
                    if (gate && !_lastTTL)
                    {
                        // Transpose channel-major [C x T] -> time-major [1, T, C]
                        fixed (float* pDst = inBuf)
                        {
                            byte* src  = (byte*)mat.Data.ToPointer();
                            int   step = mat.Step;
                            for (int c = 0; c < Channels; c++)
                            {
                                float* row = (float*)(src + c * step);
                                for (int t = 0; t < TimePoints; t++)
                                    pDst[t * Channels + c] = row[t];
                            }
                        }
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
                    result.Skipped = skippedFrames + (_lastTTL ? 1 : 0); // dropped during hiccup + refractory skips
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

        public void Dispose() => _cts?.Cancel();
    }
}
