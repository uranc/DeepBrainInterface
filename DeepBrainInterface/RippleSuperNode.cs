using Bonsai;
using DeepBrainInterface.Core;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace DeepBrainInterface
{
    public enum PreprocessingKind { None, ZScore, Ema }
    public enum DecisionKind { Fsm, ThresholdHold }

    [Combinator]
    [Description("Closed-loop ripple detection in one node: decimation (with phases), preprocessing, windowing, ONNX inference and the " +
                 "decision stage, with no allocation on the data path after start. Input: Zip(AmplifierData Mat, Clock ulong[]), optionally " +
                 "WithLatestFrom(gate bool). Output: RippleOut per inference, as RippleStateMachineMatBool. 1.1.79.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleSuperNode : IDisposable
    {
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        const uint _MCW_DN = 0x03000000, _DN_FLUSH = 0x01000000;

        // ---- fixed at start (changes apply at the next start) ----
        [Category("1. Input")] [TypeConverter(typeof(UnidimensionalArrayConverter))]
        [Description("Comma-separated 0-based rows of the AmplifierData Mat, in the order the model expects (repeats allowed), e.g. 30,35,31,29,28,0,27,27. " +
                     "Same list as SelectChannels in the classic chain.")]
        public int[] Channels { get; set; } = { 0, 1, 2, 3, 4, 5, 6, 7 };
        [Category("1. Input")] [Description("Keep every Nth raw sample, the first one kept (12 = 30 kHz -> 2.5 kHz). 1 = raw rate. Same as Buffer(1, N).")]
        public int DecimationFactor { get; set; } = 12;
        [Category("1. Input")]
        [Description("How many of the DecimationFactor sample offsets are processed. 1 = plain decimation, exactly the classic chain. " +
                     "P > 1 keeps P shifted copies of the decimated signal (offsets 0..P-1 raw samples), each with its own preprocessing state and " +
                     "its own [TimePoints x Channels] window; every Stride the P windows are run as a batch [P, TimePoints, Channels] when the ONNX " +
                     "batch dimension is dynamic or equals P, otherwise as P separate [1, TimePoints, Channels] calls (the deployed models are batch 1, " +
                     "so P times the inference cost). The P probabilities are reduced by Combiner before the decision stage. Phase 0 sets the cadence.")]
        public int PhaseCount { get; set; } = 1;

        [Category("2. Preprocessing")]
        [Description("ZScore = deployed: per-channel sliding z-score over ZScoreWindow kept samples (bit-identical to SlidingWindowZScore). " +
                     "Ema = experimental exponential mean/variance (EmaAlpha), NOT what the deployed models were trained with. None = raw values.")]
        public PreprocessingKind Preprocessing { get; set; } = PreprocessingKind.ZScore;
        [Category("2. Preprocessing")] [Description("ZScore: window length in kept samples (1250 = 0.5 s at 2.5 kHz).")]
        public int ZScoreWindow { get; set; } = 1250;
        [Category("2. Preprocessing")] [Description("Added to the standard deviation (both kinds).")]
        public float ZScoreEps { get; set; } = 1e-8f;
        [Category("2. Preprocessing")] [Description("Ema only: update weight per kept sample; the time constant is 1/alpha samples (0.01 = 100 samples = 40 ms at 2.5 kHz). Ignored for ZScore.")]
        public float EmaAlpha { get; set; } = 0.01f;

        [Category("3. Model")] [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        [Description("ONNX model, input [batch, TimePoints, Channels], output [batch, 1] probability. Model 0.")]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";
        [Category("3. Model")] [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        [Description("OPTIONAL second model (model 1), same shapes. Leave empty to run one model, which is the deployed setup; ActiveModels must then stay 0.")]
        public string ModelPath2 { get; set; } = "";
        [Category("3. Model")] [Description("Window length in kept samples (44 for the deployed models).")] public int TimePoints { get; set; } = 44;
        [Category("3. Model")] [Description("Inference every Stride kept samples (2 = every 0.8 ms at 2.5 kHz). Same as Buffer(TimePoints, Stride).")] public int Stride { get; set; } = 2;

        [Category("4. Decision")]
        [Description("Fsm = the deployed RippleStateMachineMatBool logic (groups 5a-5d). ThresholdHold = minimal alternative (group 4b): TTL when the probability " +
                     "stays above HoldThreshold for HoldConsecutive decisions, held for PostRippleMs.")]
        public DecisionKind Decision { get; set; } = DecisionKind.Fsm;
        [Category("4b. Decision: ThresholdHold only")] [Description("Probability threshold.")] public float HoldThreshold { get; set; } = 0.5f;
        [Category("4b. Decision: ThresholdHold only")] [Description("Consecutive decisions above HoldThreshold before the TTL.")] public int HoldConsecutive { get; set; } = 3;
        [Category("4c. Decision: timing")]
        [Description("false = hold, delay and decay grace are timed on the wall clock (deployed behaviour). true = timed on the headstage clock carried with " +
                     "each decision (sample time): a stalled thread cannot stretch a hold, and the times match the Clock column of the CSV exactly. " +
                     "Same cost either way.")]
        public bool UseHardwareClock { get; set; } = false;
        [Category("4c. Decision: timing")] [Description("Headstage clock rate (ONIX: 250e6).")] public double HardwareClockHz { get; set; } = 250e6;

        [Category("6. Snapshot")] [Description("Attach the model input window ([TimePoints x Channels] after preprocessing, reused array) to RippleOut on each TTL rise. " +
                                               "Not written by the CSV writer unless SnapshotWindow is added to its selector.")]
        public bool SnapshotEnabled { get; set; } = false;

        // ---- live ----
        int activeModels; Combiner combiner;
        [Category("3. Model")] [Description("0 = model 0 only (deployed, the only valid value without ModelPath2), 1 = model 1 only, 2 = both, combined by Combiner. Live.")]
        public int ActiveModels { get { return activeModels; } set { activeModels = value; var p = proc; if (p != null) p.SetActiveModels(value); } }
        [Category("3. Model")] [Description("Reduction over the PhaseCount phases and, with ActiveModels = 2, over the two models: First, Max or Mean. Live.")]
        public Combiner Combiner { get { return combiner; } set { combiner = value; var p = proc; if (p != null) p.SetCombiner(value); } }

        readonly RippleFsm fsm = new RippleFsm();
        [Category("5. FSM: General")] [Description("Master gate. false = the FSM stays in NoRipple and never triggers. Live.")]
        public bool DetectionEnabled { get { return fsm.DetectionEnabled; } set { fsm.DetectionEnabled = value; fsm.ConfigVersion++; } }
        [Category("5a. FSM: Thresholds (0.0 to 1.0)")] [Description("Below this the FSM ignores the frame (NoRipple). Live.")]
        public float Threshold1_IgnoreBelow { get { return fsm.Threshold1; } set { fsm.Threshold1 = value; fsm.ConfigVersion++; } }
        [Category("5a. FSM: Thresholds (0.0 to 1.0)")] [Description("Weak evidence: +1.0 to the score. Live.")]
        public float Threshold2_WeakEvidence { get { return fsm.Threshold2; } set { fsm.Threshold2 = value; fsm.ConfigVersion++; } }
        [Category("5a. FSM: Thresholds (0.0 to 1.0)")] [Description("Strong evidence: +2.5 to the score. Live.")]
        public float Threshold3_StrongEvidence { get { return fsm.Threshold3; } set { fsm.Threshold3 = value; fsm.ConfigVersion++; } }
        [Category("5b. FSM: Scoring")] [Description("Score that triggers the event. Live.")]
        public float TargetEvidenceScore { get { return fsm.TargetEvidenceScore; } set { fsm.TargetEvidenceScore = value; fsm.ConfigVersion++; } }
        [Category("5c. FSM: Leaky Bucket (Anti-Flicker)")] [Description("Decisions below Threshold1 tolerated before the score starts to decay. Live.")]
        public int FramesToWaitBeforeDecay { get { return fsm.FramesToWaitBeforeDecay; } set { fsm.FramesToWaitBeforeDecay = value; fsm.ConfigVersion++; } }
        [Category("5c. FSM: Leaky Bucket (Anti-Flicker)")]
        [Description("Same grace in milliseconds of the hardware clock (about: whole-ms resolution), used instead of FramesToWaitBeforeDecay when UseHardwareClock is true and this is > 0. Live.")]
        public int DecayWaitMs { get { return fsm.DecayWaitMs; } set { fsm.DecayWaitMs = value; fsm.ConfigVersion++; } }
        [Category("5c. FSM: Leaky Bucket (Anti-Flicker)")] [Description("Score subtracted per decision once the grace is over. Live.")]
        public float ScoreDecayPerFrame { get { return fsm.ScoreDecayPerFrame; } set { fsm.ScoreDecayPerFrame = value; fsm.ConfigVersion++; } }
        [Category("5d. FSM: TTL Output")] [Description("'0' = TTL at the trigger; '250' = fixed delay in ms; '[150, 250]' = uniform random delay per event. Live.")]
        public string TriggerDelayMs { get { return fsm.TriggerDelaySetting; } set { fsm.TriggerDelaySetting = value; int dmin, dmax; RippleStateMachineMatBool.ParseDelay(value, out dmin, out dmax); fsm.SetDelay(dmin, dmax); } }
        [Category("5d. FSM: TTL Output")] [Description("TTL hold / refractory period in ms. Live.")]
        public int PostRippleMs { get { return fsm.PostRippleMs; } set { fsm.PostRippleMs = value; fsm.ConfigVersion++; } }

        volatile FrameProcessor proc;
        int subscribed;
        bool fpuSet;

        FrameProcessor Build()
        {
            var cfg = new ProcessorConfig
            {
                Channels = (int[])Channels.Clone(), DecimationFactor = DecimationFactor, PhaseCount = PhaseCount, TimePoints = TimePoints, Stride = Stride,
                ModelPaths = string.IsNullOrWhiteSpace(ModelPath2) ? new[] { ModelPath } : new[] { ModelPath, ModelPath2 },
                ActiveModels = activeModels, Combiner = combiner, SnapshotEnabled = SnapshotEnabled,
            };
            int ch = cfg.Channels.Length; var kind = Preprocessing; int zw = ZScoreWindow; float eps = ZScoreEps, alpha = EmaAlpha;
            cfg.StageFactory = p =>
            {
                switch (kind)
                {
                    case PreprocessingKind.ZScore: return new IPreprocessStage[] { new ZScoreStage(ch, zw, eps) };
                    case PreprocessingKind.Ema: return new IPreprocessStage[] { new EmaStage(ch, alpha, eps) };
                    default: return new IPreprocessStage[0];
                }
            };
            IDecisionStage stage;
            if (Decision == DecisionKind.ThresholdHold) stage = new ThresholdHoldStage(HoldThreshold, HoldConsecutive, PostRippleMs);
            else { fsm.HardwareClockHz = UseHardwareClock ? HardwareClockHz : 0; fsm.Reset(); stage = fsm; }
            return new FrameProcessor(cfg, stage);
        }

        /// <summary>One subscription per node instance; the processor is created on the first frame, on the data thread
        /// (ONNX Runtime and the FPU denormal flag bind to the initialising thread).</summary>
        IObservable<RippleOut> Run<T>(IObservable<T> source, Func<T, Mat> mat, Func<T, ulong[]> clock, Func<T, bool> gate)
        {
            return Observable.Create<RippleOut>(observer =>
            {
                if (Interlocked.CompareExchange(ref subscribed, 1, 0) != 0)
                {
                    observer.OnError(new InvalidOperationException("RippleSuperNode allows one subscription per node instance (second subscription refused)"));
                    return System.Reactive.Disposables.Disposable.Empty;
                }
                var outBuf = new RippleOut[0];
                var sub = source.Subscribe(item =>
                {
                    try
                    {
                        var p = proc;
                        if (p == null)
                        {
                            if (!fpuSet) { try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { } fpuSet = true; }
                            p = Build(); proc = p;
                        }
                        Mat m = mat(item); ulong[] ck = clock(item);
                        if (m == null || ck == null) return;
                        if (ck.Length < m.Cols) throw new InvalidOperationException($"clock array has {ck.Length} entries for {m.Cols} samples");
                        if (outBuf.Length < m.Cols) outBuf = new RippleOut[m.Cols];
                        int n;
                        unsafe
                        {
                            fixed (ulong* pc = ck)
                            {
                                switch (m.Depth)
                                {
                                    case Depth.U16: n = p.ProcessFrame((ushort*)m.Data.ToPointer(), m.Rows, m.Cols, m.Step, pc, gate(item), outBuf); break;
                                    case Depth.S16: n = p.ProcessFrameS16((short*)m.Data.ToPointer(), m.Rows, m.Cols, m.Step, pc, gate(item), outBuf); break;
                                    case Depth.F32: n = p.ProcessFrameF32((float*)m.Data.ToPointer(), m.Rows, m.Cols, m.Step, pc, gate(item), outBuf); break;
                                    default: throw new InvalidOperationException("RippleSuperNode: input depth must be U16, S16 or F32, got " + m.Depth);
                                }
                            }
                        }
                        for (int i = 0; i < n; i++) observer.OnNext(outBuf[i]);
                    }
                    catch (Exception ex) { observer.OnError(ex); }
                }, observer.OnError, observer.OnCompleted);
                return System.Reactive.Disposables.Disposable.Create(() => { sub.Dispose(); var p = proc; proc = null; p?.Dispose(); fpuSet = false; Interlocked.Exchange(ref subscribed, 0); });
            });
        }

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong[]>> source)
            => Run(source, t => t.Item1, t => t.Item2, t => true);

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong[]>, bool>> source)
            => Run(source, t => t.Item1.Item1, t => t.Item1.Item2, t => t.Item2);

        public void Dispose() { var p = proc; proc = null; p?.Dispose(); }
    }
}
