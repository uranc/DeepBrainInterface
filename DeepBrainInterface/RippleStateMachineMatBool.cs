using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    // ==============================================================================
    // SHARED TYPES
    // ==============================================================================
    public enum RippleState { NoRipple, Possible, Ripple }

    public sealed class RippleOut
    {
        public RippleState State { get; set; }
        public float Probability { get; set; }
        public int StrideUsed { get; set; }
        public int EventCount { get; set; }

        public float Score { get; set; }
        public float LastEventScore { get; set; }

        public bool EventPulse { get; set; }
        public bool TriggerPulse { get; set; }
        public bool TTL { get; set; }

        public Mat TriggerData { get; set; }
    }

    [Combinator]
    [Description("Logic Engine: Probability + BNO Gate → RippleState.")]
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

        // ---- Timing ----
        [Category("TTL"), DisplayName("Trigger Delay (ms)")]
        public int TriggerDelayMs { get; set; } = 0;

        [Category("TTL"), DisplayName("PostRipple Hold (ms)")]
        public int PostRippleMs { get; set; } = 50;

        // ---- Internal ----
        RippleState _state = RippleState.NoRipple;
        int _scoreTicks;
        int _eventCount;
        float _lastEventScore;

        bool _ttlArmed;
        long _ttlAtMs;
        bool _ttlHolding;
        long _ttlHoldUntilMs;

        static readonly Stopwatch Clock = Stopwatch.StartNew();

        public RippleOut Update(float prob, bool bnoOk, Mat inputData)
        {
            long now = Clock.ElapsedMilliseconds;
            bool trigger = false;
            Mat triggerSnapshot = null;

            // A. TTL Hold
            if (_ttlHolding)
            {
                if (now >= _ttlHoldUntilMs)
                {
                    _ttlHolding = false;
                    _state = RippleState.NoRipple;
                    _scoreTicks = 0;
                }
                else
                {
                    return Pack(prob, false, true, null);
                }
            }

            // B. Delayed Trigger
            if (_ttlArmed && now >= _ttlAtMs)
            {
                _ttlArmed = false;
                _ttlHolding = true;
                _ttlHoldUntilMs = now + Math.Max(0, PostRippleMs);
                trigger = true;
                triggerSnapshot = inputData?.Clone();
                return Pack(prob, true, true, triggerSnapshot);
            }

            // C. Detection Logic
            bool gatesOn = DetectionEnabled && bnoOk;
            if (!gatesOn)
            {
                _state = RippleState.NoRipple;
                _scoreTicks = 0;
                return Pack(prob, false, false, null);
            }

            int eventTicks = Math.Max(1, (int)Math.Ceiling(EventScoreThreshold * 2f));

            switch (_state)
            {
                case RippleState.NoRipple:
                    if (prob >= GateThreshold)
                    {
                        _state = RippleState.Possible;
                        _scoreTicks = 0;
                    }
                    break;

                case RippleState.Possible:
                    if (prob >= ConfirmThreshold)
                    {
                        if (prob >= ConfirmThreshold) _scoreTicks += 2;
                        else if (prob >= EnterThreshold) _scoreTicks += 1;

                        if (_scoreTicks >= eventTicks)
                        {
                            _state = RippleState.Ripple;
                            _eventCount++;
                            _lastEventScore = _scoreTicks * 0.5f;
                            _scoreTicks = 0;

                            if (TriggerDelayMs > 0)
                            {
                                _ttlArmed = true;
                                _ttlAtMs = now + TriggerDelayMs;
                            }
                            else
                            {
                                trigger = true;
                                triggerSnapshot = inputData?.Clone();
                                _ttlHolding = true;
                                _ttlHoldUntilMs = now + Math.Max(0, PostRippleMs);
                            }
                        }
                    }
                    else if (prob < GateThreshold)
                    {
                        _state = RippleState.NoRipple;
                        _scoreTicks = 0;
                    }
                    break;

                case RippleState.Ripple:
                    // Exit Logic: Drop below Gate
                    if (prob < GateThreshold)
                    {
                        _state = RippleState.NoRipple;
                        _scoreTicks = 0;
                    }
                    break;
            }

            return Pack(prob, trigger, _ttlHolding || _ttlArmed, triggerSnapshot);
        }

        private RippleOut Pack(float d, bool pulse, bool ttl, Mat data)
        {
            return new RippleOut
            {
                State = _state,
                Score = _scoreTicks * 0.5f,
                Probability = d,
                EventCount = _eventCount,
                LastEventScore = _lastEventScore,
                EventPulse = pulse,
                TriggerPulse = pulse,
                TTL = ttl,
                TriggerData = data,
                StrideUsed = 0
            };
        }

        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            return source.Select(m =>
            {
                float val = 0;
                if (m.Depth == Depth.F32) unsafe { val = *((float*)m.Data.ToPointer()); }
                return Update(val, true, null);
            });
        }

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
        {
            return source.Select(t =>
            {
                float val = 0;
                if (t.Item1.Depth == Depth.F32) unsafe { val = *((float*)t.Item1.Data.ToPointer()); }
                return Update(val, t.Item2, null);
            });
        }
    }
}