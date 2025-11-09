using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    public enum RippleState { NoRipple, Possible, Ripple }

    public sealed class RippleIn
    {
        public float Prob { get; set; }    // model probability 0..1
        public bool BnoGate { get; set; } // thresholded BNO gate (bool)
        public int Skip { get; set; }    // step size indicator (>=1); if <=0, DefaultSkip is used
    }

    public sealed class RippleOut
    {
        public RippleState State { get; set; }
        public float Score { get; set; }            // running consecutive score
        public float DecisionValue { get; set; }    // == Prob (no baseline)
        public int Skip { get; set; }             // pass-through / clamped
        public int EventCount { get; set; }       // total confirmed events
        public float LastEventScore { get; set; }   // score at last confirmation
    }

    [Description("Gate→Possible→Ripple with simple scoring (Enter=+0.5, Confirm=+1). No baseline, no refractory.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class RippleStateMachine : Transform<RippleIn, RippleOut>
    {
        // --- Top-level controls ---
        [Category("General")]
        [DisplayName("Detection Enabled")]
        [Description("When true, detection is active.")]
        public bool DetectionEnabled { get; set; } = true;

        [Category("General")]
        [DisplayName("Default Skip")]
        [Description("Used if input Skip <= 0.")]
        public int DefaultSkip { get; set; } = 1;

        // --- Thresholds in crossing order ---
        [Category("Thresholds (crossing order)")]
        [DisplayName("1. Gate (arm)")]
        [Description("Arm Possible when Prob ≥ this value.")]
        public float GateThreshold { get; set; } = 0.10f;

        [Category("Thresholds (crossing order)")]
        [DisplayName("2. Enter (+0.5 score)")]
        [Description("Adds +0.5 to Score when Prob ≥ this value.")]
        public float EnterThreshold { get; set; } = 0.50f;

        [Category("Thresholds (crossing order)")]
        [DisplayName("3. Confirm (+1.0 score)")]
        [Description("Adds +1.0 to Score when Prob ≥ this value; sustains Ripple.")]
        public float ConfirmThreshold { get; set; } = 0.80f;

        [Category("Thresholds (crossing order)")]
        [DisplayName("4. Event Score (> triggers)")]
        [Description("Event fires when Score exceeds this value.")]
        public float EventScoreThreshold { get; set; } = 2.5f;

        // ---- Internal ----
        RippleState _state = RippleState.NoRipple;
        float _score;
        int _eventCount;
        float _lastEventScore;

        public override IObservable<RippleOut> Process(IObservable<RippleIn> source)
        {
            return source.Select(Tick);
        }

        RippleOut Tick(RippleIn inp)
        {
            if (inp == null) inp = new RippleIn();

            // Clamp skip and thresholds
            int skip = inp.Skip > 0 ? inp.Skip : Math.Max(1, DefaultSkip);
            float gate = Clamp01(GateThreshold);
            float enter = Clamp01(EnterThreshold);
            float confirm = Clamp01(ConfirmThreshold);

            bool gatesOn = DetectionEnabled && inp.BnoGate;
            float d = inp.Prob; // decision value (no baseline)

            if (!gatesOn)
            {
                _state = RippleState.NoRipple;
                _score = 0f;
            }
            else
            {
                switch (_state)
                {
                    case RippleState.NoRipple:
                        if (d >= gate)
                        {
                            _state = RippleState.Possible;
                            _score = 0f;
                        }
                        break;

                    case RippleState.Possible:
                        if (d < gate)
                        {
                            _state = RippleState.NoRipple;
                            _score = 0f;
                        }
                        else
                        {
                            if (d >= confirm) _score += 1.0f;
                            else if (d >= enter) _score += 0.5f;
                            else _score = 0f;

                            if (_score > EventScoreThreshold)
                            {
                                _state = RippleState.Ripple;
                                _eventCount++;
                                _lastEventScore = _score;
                            }
                        }
                        break;

                    case RippleState.Ripple:
                        if (d >= confirm)
                        {
                            // stay Ripple while strong
                        }
                        else if (d >= gate)
                        {
                            _state = RippleState.Possible;
                            _score = 0f;
                        }
                        else
                        {
                            _state = RippleState.NoRipple;
                            _score = 0f;
                        }
                        break;
                }
            }

            return new RippleOut
            {
                State = _state,
                Score = _score,
                DecisionValue = d,
                Skip = skip,
                EventCount = _eventCount,
                LastEventScore = _lastEventScore
            };
        }

        static float Clamp01(float v)
        {
            if (v < 0f) return 0f;
            if (v > 1f) return 1f;
            return v;
        }
    }
}
