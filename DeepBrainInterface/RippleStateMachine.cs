using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    // Original names, defined here so this file compiles on its own
    public enum RippleState { NoRipple, Possible, Ripple }

    public sealed class RippleOut
    {
        public RippleState State { get; set; }
        public float Score { get; set; }            // half-step ticks × 0.5
        public float DecisionValue { get; set; }    // prob 0..1
        public int Skip { get; set; }             // suggested K (telemetry)
        public int EventCount { get; set; }
        public float LastEventScore { get; set; }
        public bool EventPulse { get; set; }       // one tick on Ripple entry
        public bool TTL { get; set; }              // true while Ripple
    }

    [Description("FSM for (prob Mat, BNO bool). Half-step scoring; outputs RippleOut.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class RippleStateMachineMatBool : Transform<Tuple<Mat, bool>, RippleOut>
    {
        // ---- General ----
        [Category("General"), DisplayName("Detection Enabled")]
        public bool DetectionEnabled { get; set; } = true;

        // ---- Thresholds ----
        [Category("Thresholds"), DisplayName("1. Gate (arm)")]
        public float GateThreshold { get; set; } = 0.10f;

        [Category("Thresholds"), DisplayName("2. Enter (+0.5 per tick)")]
        public float EnterThreshold { get; set; } = 0.50f;

        [Category("Thresholds"), DisplayName("3. Confirm (+1.0 per tick)")]
        public float ConfirmThreshold { get; set; } = 0.80f;

        [Category("Thresholds"), DisplayName("4. Event Score (≥ triggers)")]
        [Description("Half-steps: +0.5 when ≥Enter; +1.0 when ≥Confirm.")]
        public float EventScoreThreshold { get; set; } = 2.5f;

        // ---- Suggested stride K per band (telemetry; drive your stride node) ----
        [Category("Stride K"), DisplayName("K below Gate (< Gate)")]
        public int KBelowGate { get; set; } = 4;

        [Category("Stride K"), DisplayName("K at Gate (Gate ≤ d < Enter)")]
        public int KAtGate { get; set; } = 2;

        [Category("Stride K"), DisplayName("K at Enter (d ≥ Enter)")]
        public int KAtEnter { get; set; } = 1;

        // ---- Internal state ----
        RippleState _state = RippleState.NoRipple;
        RippleState _prev = RippleState.NoRipple;
        int _scoreTicks;            // integer half-steps (1 = 0.5)
        int _eventCount;
        float _lastEventScore;

        public override IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
            => source.Select(x => Tick(ReadProb(x.Item1), x.Item2));

        RippleOut Tick(float prob, bool bnoOk)
        {
            float gate = Clamp01(GateThreshold);
            float enter = Clamp01(EnterThreshold);
            float confirm = Clamp01(ConfirmThreshold);
            int eventTicks = (int)Math.Ceiling(EventScoreThreshold * 2f); // half-steps

            bool gatesOn = DetectionEnabled && bnoOk;
            float d = prob;
            bool eventPulse = false;

            _prev = _state;

            if (!gatesOn)
            {
                _state = RippleState.NoRipple;
                _scoreTicks = 0;
            }
            else
            {
                switch (_state)
                {
                    case RippleState.NoRipple:
                        if (d >= gate) { _state = RippleState.Possible; _scoreTicks = 0; }
                        break;

                    case RippleState.Possible:
                        if (d < gate)
                        {
                            _state = RippleState.NoRipple;
                            _scoreTicks = 0;
                        }
                        else
                        {
                            if (d >= confirm) _scoreTicks += 2;   // +1.0
                            else if (d >= enter) _scoreTicks += 1;   // +0.5
                            else _scoreTicks = 0;   // reset when < Enter

                            if (_scoreTicks >= eventTicks)
                            {
                                _state = RippleState.Ripple;
                                _eventCount++;
                                _lastEventScore = _scoreTicks * 0.5f;
                                _scoreTicks = 0;
                                eventPulse = true; // one tick on entry
                            }
                        }
                        break;

                    case RippleState.Ripple:
                        if (d >= confirm) { /* stay */ }
                        else if (d >= gate) { _state = RippleState.Possible; _scoreTicks = 0; }
                        else { _state = RippleState.NoRipple; _scoreTicks = 0; }
                        break;
                }
            }

            // Suggested K (telemetry)
            int kOut = (!gatesOn || d < gate) ? KBelowGate : (d < enter ? KAtGate : KAtEnter);
            if (kOut < 0) kOut = 0;

            return new RippleOut
            {
                State = _state,
                Score = _scoreTicks * 0.5f,
                DecisionValue = d,
                Skip = kOut,
                EventCount = _eventCount,
                LastEventScore = _lastEventScore,
                EventPulse = eventPulse,
                TTL = (_state == RippleState.Ripple)
            };
        }

        static float ReadProb(Mat m)
        {
            if (m == null) return 0f;
            if (m.Depth != Depth.F32) throw new ArgumentException("Expected F32 Mat (prob).");
            unsafe { return *((float*)m.Data.ToPointer()); } // read first float (1×1 or first channel)
        }

        static float Clamp01(float v) { if (v < 0f) return 0f; if (v > 1f) return 1f; return v; }
    }
}
