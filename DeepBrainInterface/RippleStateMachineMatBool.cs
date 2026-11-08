using Bonsai;
using DeepBrainInterface.Core;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Accumulator FSM. Thread-safe with Leaky Bucket Grace Period. Shell over DeepBrainInterface.Core.RippleFsm (1.1.78).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class RippleStateMachineMatBool
    {
        readonly RippleFsm fsm = new RippleFsm();
        readonly float[] probs = new float[1];
        readonly object fsmLock = new object();

        [Category("General")] public bool DetectionEnabled { get { return fsm.DetectionEnabled; } set { fsm.DetectionEnabled = value; fsm.ConfigVersion++; } }

        [Category("1. Thresholds (0.0 to 1.0)")]
        [Description("Threshold 1 (Floor): If probability drops below this, start the dip timer.")]
        public float Threshold1_IgnoreBelow { get { return fsm.Threshold1; } set { fsm.Threshold1 = value; fsm.ConfigVersion++; } }

        [Category("1. Thresholds (0.0 to 1.0)")]
        [Description("Threshold 2 (Weak): Awards 1 evidence point per frame.")]
        public float Threshold2_WeakEvidence { get { return fsm.Threshold2; } set { fsm.Threshold2 = value; fsm.ConfigVersion++; } }

        [Category("1. Thresholds (0.0 to 1.0)")]
        [Description("Threshold 3 (Strong): Awards 2.5 evidence points per frame.")]
        public float Threshold3_StrongEvidence { get { return fsm.Threshold3; } set { fsm.Threshold3 = value; fsm.ConfigVersion++; } }

        [Category("2. Scoring")]
        [Description("Total points needed to trigger the hardware TTL.")]
        public float TargetEvidenceScore { get { return fsm.TargetEvidenceScore; } set { fsm.TargetEvidenceScore = value; fsm.ConfigVersion++; } }

        [Category("3. Leaky Bucket (Anti-Flicker)")]
        [Description("How many consecutive frames to WAIT during a dip BEFORE subtracting points.")]
        public int FramesToWaitBeforeDecay { get { return fsm.FramesToWaitBeforeDecay; } set { fsm.FramesToWaitBeforeDecay = value; fsm.ConfigVersion++; } }

        [Category("3. Leaky Bucket (Anti-Flicker)")]
        [Description("How many points to subtract from the score per frame AFTER the wait period.")]
        public float ScoreDecayPerFrame { get { return fsm.ScoreDecayPerFrame; } set { fsm.ScoreDecayPerFrame = value; fsm.ConfigVersion++; } }

        [Category("4. TTL Output")]
        [Description("Delay in milliseconds before triggering the TTL. Enter a single number for a fixed delay (e.g., '250') or a bracketed range for a uniform random delay (e.g., '[250, 400]').")]
        public string TriggerDelayMs
        {
            get { return fsm.TriggerDelaySetting; }
            set
            {
                fsm.TriggerDelaySetting = value;
                int dmin, dmax; ParseDelay(value, out dmin, out dmax); fsm.SetDelay(dmin, dmax);
            }
        }

        [Category("4. TTL Output")] public int PostRippleMs { get { return fsm.PostRippleMs; } set { fsm.PostRippleMs = value; fsm.ConfigVersion++; } }

        /// <summary>"250" -> (250, 250); "[250, 400]" -> (250, 400); blank -> (0, 0). Shared with RippleSuperNode.</summary>
        public static void ParseDelay(string value, out int min, out int max)
        {
            min = max = 0;
            if (string.IsNullOrWhiteSpace(value)) return;
            var parts = value.Trim('[', ']').Split(',');
            min = (int)float.Parse(parts[0]);
            max = parts.Length > 1 ? (int)float.Parse(parts[1]) : min;
            if (min > max) { int t = min; min = max; max = t; }
        }

        public RippleOut Update(float signal, float artProb, bool gateOpen, Mat rawInput)
        {
            lock (fsmLock) { return fsm.Update(signal, artProb, gateOpen); }
        }

        private float GetVal(Mat m)
        {
            if (m == null) return 0f;
            unsafe { return *((float*)m.Data.ToPointer()); }
        }

        void Reset() { lock (fsmLock) fsm.Reset(); }

        // Every overload resets the FSM state at subscribe (workflow start) via Observable.Defer.

        public IObservable<RippleOut> Process(IObservable<Mat> source)
            => Observable.Defer(() => { Reset(); return source.Select(m => Update(GetVal(m), 0f, true, null)); });

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
            => Observable.Defer(() => { Reset(); return source.Select(t => Update(GetVal(t.Item1), 0f, t.Item2, null)); });

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong>> source)
            => Observable.Defer(() => { Reset(); return source.Select(t => { var r = Update(GetVal(t.Item1), 0f, true, null); r.Clock = t.Item2; return r; }); });

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong[]>> source)
            => Observable.Defer(() => { Reset(); return source.Select(t => { var r = Update(GetVal(t.Item1), 0f, true, null); r.Clock = (t.Item2 != null && t.Item2.Length > 0) ? t.Item2[t.Item2.Length - 1] : 0UL; return r; }); });

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong>, bool>> source)
            => Observable.Defer(() => { Reset(); return source.Select(t => { var r = Update(GetVal(t.Item1.Item1), 0f, t.Item2, null); r.Clock = t.Item1.Item2; return r; }); });

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong[]>, bool>> source)
            => Observable.Defer(() => { Reset(); return source.Select(t => { var r = Update(GetVal(t.Item1.Item1), 0f, t.Item2, null); r.Clock = (t.Item1.Item2 != null && t.Item1.Item2.Length > 0) ? t.Item1.Item2[t.Item1.Item2.Length - 1] : 0UL; return r; }); });

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
            => Observable.Defer(() => { Reset(); return source.Select(t => Update(GetVal(t.Item1.Item1), 0f, t.Item2, t.Item1.Item2)); });
    }
}
