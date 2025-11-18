using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    public enum RippleState { NoRipple, Possible, Ripple }

    public sealed class RippleOut
    {
        public RippleState State { get; set; }
        public float Score { get; set; }           // half-step ticks × 0.5
        public float DecisionValue { get; set; }   // prob 0..1
        public int Skip { get; set; }              // suggested K
        public int EventCount { get; set; }
        public float LastEventScore { get; set; }
        public bool EventPulse { get; set; }       // one tick at TTL rising edge
        public bool TTL { get; set; }              // HIGH during PostRipple hold
    }

    [Combinator]
    [Description("Prob (Mat) with optional BNO bool gate → RippleOut. TTL rising-edge delay + fixed hold.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class RippleStateMachineMatBool
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
        public float EventScoreThreshold { get; set; } = 2.5f;

        // ---- Suggested stride K (telemetry) ----
        [Category("Stride K"), DisplayName("K below Gate (< Gate)")]
        public int KBelowGate { get; set; } = 4;
        [Category("Stride K"), DisplayName("K at Gate (Gate ≤ d < Enter)")]
        public int KAtGate { get; set; } = 2;
        [Category("Stride K"), DisplayName("K at Enter (d ≥ Enter)")]
        public int KAtEnter { get; set; } = 1;

        // ---- Timing ----
        [Category("TTL"), DisplayName("Trigger Delay (ms)")]
        public int TriggerDelayMs { get; set; } = 0;

        [Category("TTL"), DisplayName("PostRipple Hold (ms)")]
        public int PostRippleMs { get; set; } = 50;

        // ---- Internal ----
        RippleState _state = RippleState.NoRipple;
        int _scoreTicks; // 1 tick = 0.5f
        int _eventCount;
        float _lastEventScore;

        bool _ttlArmed;
        long _ttlAtMs;
        bool _ttlHolding;
        long _ttlHoldUntilMs;

        static readonly Stopwatch Clock = Stopwatch.StartNew();

        // ========= Process overloads =========

        // A) Prob only (no BNO wired): BNO assumed false (gate closed).
        public IObservable<RippleOut> Process(IObservable<Mat> probMat)
        {
            return probMat.Select(m => Tick(ReadProb(m), false));
        }

        // B) Prob + BNO bool
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> probAndBno)
        {
            return probAndBno.Select(x => Tick(ReadProb(x.Item1), x.Item2));
        }

        // ========= Core tick =========
        RippleOut Tick(float prob, bool bnoOk)
        {
            float gate = Clamp01(GateThreshold);
            float enter = Clamp01(EnterThreshold);
            float confirm = Clamp01(ConfirmThreshold);
            int eventTicks = Math.Max(1, (int)Math.Ceiling(EventScoreThreshold * 2f));
            float d = prob;
            long now = Clock.ElapsedMilliseconds;

            // TTL hold window (non-blocking)
            if (_ttlHolding)
            {
                if (now >= _ttlHoldUntilMs)
                {
                    _ttlHolding = false;
                    _state = RippleState.NoRipple;
                    _scoreTicks = 0;
                    return Pack(d, KBelowGate, false, false);
                }
                return Pack(d, KBelowGate, false, true);
            }

            // Delayed rising edge
            if (_ttlArmed && now >= _ttlAtMs)
            {
                _ttlArmed = false;
                _ttlHolding = true;
                _ttlHoldUntilMs = now + Math.Max(0, PostRippleMs);
                return Pack(d, KBelowGate, true, true); // EventPulse=1 on rise
            }

            // FSM (gated)
            bool gatesOn = DetectionEnabled && bnoOk;
            if (!gatesOn)
            {
                _ttlArmed = false;
                _state = RippleState.NoRipple;
                _scoreTicks = 0;
                return Pack(d, kBand(d, gate, enter, gatesOn), false, false);
            }

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
                        else if (d >= enter) _scoreTicks += 1; // +0.5
                        else _scoreTicks = 0;

                        if (_scoreTicks >= eventTicks)
                        {
                            _state = RippleState.Ripple;
                            _eventCount++;
                            _lastEventScore = _scoreTicks * 0.5f;
                            _scoreTicks = 0;

                            _ttlArmed = true;
                            _ttlAtMs = now + Math.Max(0, TriggerDelayMs);
                        }
                    }
                    break;

                case RippleState.Ripple:
                    if (d >= confirm) { /* stay */ }
                    else if (d >= gate) { _state = RippleState.Possible; _scoreTicks = 0; }
                    else { _state = RippleState.NoRipple; _scoreTicks = 0; }
                    break;
            }

            return Pack(d, kBand(d, gate, enter, gatesOn), false, false);
        }

        RippleOut Pack(float d, int kOut, bool eventPulse, bool ttl)
        {
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
                TTL = ttl
            };
        }

        int kBand(float d, float gate, float enter, bool gatesOn)
        {
            if (!gatesOn || d < gate) return KBelowGate;
            if (d < enter) return KAtGate;
            return KAtEnter;
        }

        static float Clamp01(float v) { if (v < 0f) return 0f; if (v > 1f) return 1f; return v; }

        static float ReadProb(Mat m)
        {
            if (m == null) return 0f;
            if (m.Depth != Depth.F32) throw new ArgumentException("Expected F32 Mat (prob).");
            unsafe { return *((float*)m.Data.ToPointer()); } // first float
        }
    }
}
