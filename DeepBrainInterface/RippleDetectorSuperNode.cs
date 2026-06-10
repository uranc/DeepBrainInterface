using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Ring-buffered inference + detection wrapper. Acquisition thread writes windowed Mats; " +
                 "background thread (pinned RealTime) runs zero-alloc ONNX + state machine, skipping inference during refractory. " +
                 "Output: identical RippleOut with Clock.")]
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

        // --- 1. Model ---
        [Category("1. Model")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";
        [Category("1. Model")] public int TimePoints { get; set; } = 44;
        [Category("1. Model")] public int Channels { get; set; } = 8;

        // --- 2. Detection (all properties exposed at top level) ---
        [Category("2. Detection")] public bool DetectionEnabled { get; set; } = true;

        [Category("2. Detection")]
        [Description("Threshold 1 (Floor): If probability drops below this, start the dip timer.")]
        public float Threshold1_IgnoreBelow { get; set; } = 0.10f;

        [Category("2. Detection")]
        [Description("Threshold 2 (Weak): Awards 1 evidence point per frame.")]
        public float Threshold2_WeakEvidence { get; set; } = 0.50f;

        [Category("2. Detection")]
        [Description("Threshold 3 (Strong): Awards 2 evidence points per frame.")]
        public float Threshold3_StrongEvidence { get; set; } = 0.80f;

        [Category("2. Detection")]
        [Description("Total points needed to trigger the hardware TTL.")]
        public float TargetEvidenceScore { get; set; } = 5.0f;

        [Category("2. Detection")]
        [Description("How many consecutive frames to WAIT during a dip BEFORE subtracting points.")]
        public int FramesToWaitBeforeDecay { get; set; } = 5;

        [Category("2. Detection")]
        [Description("How many points to subtract from the score per frame AFTER the wait period.")]
        public float ScoreDecayPerFrame { get; set; } = 1.0f;

        [Category("2. Detection")]
        [Description("Fixed delay (ms) between detection and TTL output. 0 = immediate.")]
        public int TriggerDelayMs { get; set; } = 0;

        [Category("2. Detection")]
        [Description("Randomise the trigger delay uniformly in [0, TriggerDelayMs]. For sham controls.")]
        public bool RandomizeDelay { get; set; } = false;

        [Category("2. Detection")]
        [Description("Refractory hold after a detection (ms). Inference is SKIPPED during this window.")]
        public int PostRippleMs { get; set; } = 50;

        // --- 3. Hardware ---
        [Category("3. Hardware")]
        [Description("Pin the inference thread to this core. -1 disables.")]
        public int TargetCore { get; set; } = 7;

        [Category("3. Hardware")]
        public ProcessPriorityClass ProcessPriority { get; set; } = ProcessPriorityClass.RealTime;

        // Latest windowed Mat + clock. Acquisition thread writes, background thread reads.
        private volatile Mat _currentMat;
        private volatile ulong _currentClock;
        private volatile bool _gateOpen = true;
        // Semaphore signals the background thread that a new Mat is ready.
        // Max count = 1: if inference is slower than acquisition, extra signals collapse into one.
        private SemaphoreSlim _signal;
        private CancellationTokenSource _cts;
        private RippleStateMachineMatBool _detector = new RippleStateMachineMatBool();
        private bool _lastTTL;

        private void Enqueue(Mat mat, ulong clock, bool gate)
        {
            _currentMat   = mat;
            _currentClock = clock;
            _gateOpen     = gate;
            // Only signal if not already pending — collapses rapid arrivals into one wake-up.
            // Background thread will read _currentMat which is always the latest.
            if (_signal.CurrentCount == 0) _signal.Release();
        }

        // --- Process overloads ---

        public IObservable<RippleOut> Process(IObservable<Mat> source) =>
            Run(observer => source.Subscribe(m => Enqueue(m, 0UL, true), observer.OnError));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source) =>
            Run(observer => source.Subscribe(t => Enqueue(t.Item1, 0UL, t.Item2), observer.OnError));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong>> source) =>
            Run(observer => source.Subscribe(t => Enqueue(t.Item1, t.Item2, true), observer.OnError));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong[]>> source) =>
            Run(observer => source.Subscribe(t =>
            {
                ulong clock = (t.Item2 != null && t.Item2.Length > 0) ? t.Item2[t.Item2.Length - 1] : 0UL;
                Enqueue(t.Item1, clock, true);
            }, observer.OnError));

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong>, bool>> source) =>
            Run(observer => source.Subscribe(t => Enqueue(t.Item1.Item1, t.Item1.Item2, t.Item2), observer.OnError));

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
                _currentMat = null;
                _currentClock = 0;
                _gateOpen = true;
                _lastTTL = false;
                _signal = new SemaphoreSlim(0, 1);
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

            try
            {
                while (!token.IsCancellationRequested)
                {
                    // Block until Enqueue() signals a new Mat — no spinning, no CPU waste.
                    _signal.Wait(token);

                    Mat mat = _currentMat;
                    if (mat == null) continue;

                    ulong clock = _currentClock;
                    bool gate = _gateOpen;

                    float prob = 0f;
                    if (gate && !_lastTTL)
                    {
                        // Copy Mat (channel-major [Channels x TimePoints]) into inBuf (time-major).
                        fixed (float* pDst = inBuf)
                        {
                            byte* src = (byte*)mat.Data.ToPointer();
                            int step = mat.Step;
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

                    // Sync detection properties for this frame.
                    _detector.DetectionEnabled = DetectionEnabled;
                    _detector.Threshold1_IgnoreBelow = Threshold1_IgnoreBelow;
                    _detector.Threshold2_WeakEvidence = Threshold2_WeakEvidence;
                    _detector.Threshold3_StrongEvidence = Threshold3_StrongEvidence;
                    _detector.TargetEvidenceScore = TargetEvidenceScore;
                    _detector.FramesToWaitBeforeDecay = FramesToWaitBeforeDecay;
                    _detector.ScoreDecayPerFrame = ScoreDecayPerFrame;
                    _detector.TriggerDelayMs = TriggerDelayMs;
                    _detector.RandomizeDelay = RandomizeDelay;
                    _detector.PostRippleMs = PostRippleMs;

                    RippleOut result = _detector.Update(prob, 0f, gate, null);
                    result.Clock = clock;
                    result.Skipped = _lastTTL ? 1 : 0;

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
