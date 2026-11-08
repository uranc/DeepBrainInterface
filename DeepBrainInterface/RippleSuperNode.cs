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
                 "WithLatestFrom(gate bool). Output: RippleOut per inference, as RippleStateMachineMatBool. 1.1.78.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleSuperNode : IDisposable
    {
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        const uint _MCW_DN = 0x03000000, _DN_FLUSH = 0x01000000;

        // ---- fixed at start ----
        [Category("1. Input")] [Description("0-based amplifier rows, in the order the model expects (repeats allowed).")]
        public int[] Channels { get; set; } = { 0, 1, 2, 3, 4, 5, 6, 7 };
        [Category("1. Input")] [Description("Keep every Nth raw sample (first kept). 1 = raw rate.")]
        public int DecimationFactor { get; set; } = 12;
        [Category("1. Input")] [Description("Decimation phases processed (1 = plain decimation, = DecimationFactor for all shifted copies).")]
        public int PhaseCount { get; set; } = 1;
        [Category("2. Preprocessing")] public PreprocessingKind Preprocessing { get; set; } = PreprocessingKind.ZScore;
        [Category("2. Preprocessing")] public int ZScoreWindow { get; set; } = 1250;
        [Category("2. Preprocessing")] public float ZScoreEps { get; set; } = 1e-8f;
        [Category("2. Preprocessing")] public float EmaAlpha { get; set; } = 0.01f;
        [Category("3. Model")] [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";
        [Category("3. Model")] [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))] [Description("Optional second model (empty = none).")]
        public string ModelPath2 { get; set; } = "";
        [Category("3. Model")] public int TimePoints { get; set; } = 44;
        [Category("3. Model")] [Description("Inference every Stride kept samples.")] public int Stride { get; set; } = 2;
        [Category("4. Decision")] public DecisionKind Decision { get; set; } = DecisionKind.Fsm;
        [Category("4. Decision")] [Description("ThresholdHold: probability threshold.")] public float HoldThreshold { get; set; } = 0.5f;
        [Category("4. Decision")] [Description("ThresholdHold: consecutive decisions above threshold.")] public int HoldConsecutive { get; set; } = 3;
        [Category("4. Decision")] [Description("Time the hold/delay logic on the hardware clock (250 MHz) instead of the wall clock.")] public bool UseHardwareClock { get; set; } = false;
        [Category("4. Decision")] public double HardwareClockHz { get; set; } = 250e6;
        [Category("6. Snapshot")] [Description("Attach the model input window to RippleOut on each TTL rise (reused array).")] public bool SnapshotEnabled { get; set; } = false;

        // ---- live ----
        int activeModels; Combiner combiner;
        [Category("3. Model")] [Description("0 = model 0, 1 = model 1, 2 = both (live).")]
        public int ActiveModels { get { return activeModels; } set { activeModels = value; var p = proc; if (p != null) p.SetActiveModels(value); } }
        [Category("3. Model")] [Description("Across phases and across models (live).")]
        public Combiner Combiner { get { return combiner; } set { combiner = value; var p = proc; if (p != null) p.SetCombiner(value); } }

        readonly RippleFsm fsm = new RippleFsm();
        [Category("5. FSM")] public bool DetectionEnabled { get { return fsm.DetectionEnabled; } set { fsm.DetectionEnabled = value; fsm.ConfigVersion++; } }
        [Category("5. FSM")] public float Threshold1_IgnoreBelow { get { return fsm.Threshold1; } set { fsm.Threshold1 = value; fsm.ConfigVersion++; } }
        [Category("5. FSM")] public float Threshold2_WeakEvidence { get { return fsm.Threshold2; } set { fsm.Threshold2 = value; fsm.ConfigVersion++; } }
        [Category("5. FSM")] public float Threshold3_StrongEvidence { get { return fsm.Threshold3; } set { fsm.Threshold3 = value; fsm.ConfigVersion++; } }
        [Category("5. FSM")] public float TargetEvidenceScore { get { return fsm.TargetEvidenceScore; } set { fsm.TargetEvidenceScore = value; fsm.ConfigVersion++; } }
        [Category("5. FSM")] public int FramesToWaitBeforeDecay { get { return fsm.FramesToWaitBeforeDecay; } set { fsm.FramesToWaitBeforeDecay = value; fsm.ConfigVersion++; } }
        [Category("5. FSM")] public float ScoreDecayPerFrame { get { return fsm.ScoreDecayPerFrame; } set { fsm.ScoreDecayPerFrame = value; fsm.ConfigVersion++; } }
        [Category("5. FSM")] [Description("'250' fixed or '[250, 400]' uniform random delay in ms before the TTL.")]
        public string TriggerDelayMs { get { return fsm.TriggerDelaySetting; } set { fsm.TriggerDelaySetting = value; RippleStateMachineMatBool.ParseDelay(value, out fsm.DelayMinMs, out fsm.DelayMaxMs); fsm.ConfigVersion++; } }
        [Category("5. FSM")] public int PostRippleMs { get { return fsm.PostRippleMs; } set { fsm.PostRippleMs = value; fsm.ConfigVersion++; } }

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
                return System.Reactive.Disposables.Disposable.Create(() => { sub.Dispose(); var p = proc; proc = null; p?.Dispose(); Interlocked.Exchange(ref subscribed, 0); });
            });
        }

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong[]>> source)
            => Run(source, t => t.Item1, t => t.Item2, t => true);

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong[]>, bool>> source)
            => Run(source, t => t.Item1.Item1, t => t.Item1.Item2, t => t.Item2);

        public void Dispose() { var p = proc; proc = null; p?.Dispose(); }
    }
}
