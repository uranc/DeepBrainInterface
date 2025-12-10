using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    // ==============================================================================
    // 1. DATA STRUCTURES
    // ==============================================================================
    public enum RippleState
    {
        NoRipple,
        Possible,
        Ripple
    }

    public sealed class RippleOut
    {
        // Essential Logic Outputs
        public RippleState State { get; set; }
        public bool TTL { get; set; }          // High during Hold

        // Accumulator Stats
        public float Score { get; set; }       // Continuous Score
        public int EventCount { get; set; }    // Total detected events

        // Signal Data
        public float Probability { get; set; }
        public Mat SignalData { get; set; }    // Raw Data Snapshot (Cloned)

        // Compatibility
        public int StrideUsed { get; set; }
    }

    // ==============================================================================
    // 2. STATE MACHINE NODE
    // ==============================================================================
    [Combinator]
    [Description("Accumulator FSM. Supports: (Mat), (Mat,Bool), ((Mat,Mat),Bool).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class RippleStateMachineMatBool
    {
        // ==============================================================================
        // CONFIGURATION
        // ==============================================================================

        // ---- TOP LEVEL ----
        [Category("General")]
        [Description("Master switch. If False, output is NoRipple.")]
        public bool DetectionEnabled { get; set; } = true;

        // ---- THRESHOLDS ----
        [Category("Thresholds")]
        [DisplayName("0. Gate Threshold")]
        [Description("Signal must exceed this to enter 'Possible' state.")]
        public float GateThreshold { get; set; } = 0.10f;

        [Category("Thresholds")]
        [DisplayName("1. First Threshold (Enter)")]
        [Description("Adds +1.0 score per tick if signal > this.")]
        public float EnterThreshold { get; set; } = 0.50f;

        [Category("Thresholds")]
        [DisplayName("2. Second Threshold (Confirm)")]
        [Description("Adds +2.0 score per tick if signal > this.")]
        public float ConfirmThreshold { get; set; } = 0.80f;

        [Category("Thresholds")]
        [DisplayName("3. Event Score Threshold")]
        [Description("Target Score (multiplied internally) to trigger.")]
        public float EventScoreThreshold { get; set; } = 2.5f;

        // ---- SCORE DECAY ----
        [Category("Score Decay")]
        [DisplayName("Grace Samples")]
        [Description("How long signal can hover before Score decays.")]
        public int GraceSamples { get; set; } = 5;

        [Category("Score Decay")]
        [DisplayName("Grace Rate")]
        [Description("Score subtracted per tick after Grace period.")]
        public float GraceRate { get; set; } = 1.0f;

        // ---- TTL OUTPUT ----
        [Category("TTL Output")]
        [DisplayName("Trigger Delay (ms)")]
        [Description("Delay to wait AFTER detection before raising TTL.")]
        public int TriggerDelayMs { get; set; } = 0;

        [Category("TTL Output")]
        [DisplayName("Post-Ripple Hold (ms)")]
        [Description("Duration to hold TTL High after the Delay.")]
        public int PostRippleMs { get; set; } = 50;

        // ==============================================================================
        // INTERNAL STATE
        // ==============================================================================
        RippleState _state = RippleState.NoRipple;

        float _scoreTicks;
        int _ticksInPossible;
        int _eventCount;

        bool _ttlArmed;
        long _ttlAtMs;
        bool _ttlHolding;
        long _ttlHoldUntilMs;

        static readonly Stopwatch Clock = Stopwatch.StartNew();

        // ==============================================================================
        // CORE UPDATE LOOP
        // ==============================================================================

        public RippleOut Update(float signal, bool gateOpen, Mat rawInput = null)
        {
            long now = Clock.ElapsedMilliseconds;
            Mat triggerSnapshot = null;

            float reportingScore = _scoreTicks;

            // 1. TTL HOLDING
            if (_ttlHolding)
            {
                if (now >= _ttlHoldUntilMs)
                {
                    _ttlHolding = false;
                    _state = RippleState.NoRipple;
                    _scoreTicks = 0;
                    _ticksInPossible = 0;
                    reportingScore = 0;
                }
                else
                {
                    return Pack(signal, 0f, true, null);
                }
            }

            // 2. TRIGGER DELAY
            if (_ttlArmed)
            {
                if (now >= _ttlAtMs)
                {
                    _ttlArmed = false;
                    _ttlHolding = true;
                    _ttlHoldUntilMs = now + Math.Max(1, PostRippleMs);
                    triggerSnapshot = rawInput?.Clone(); // Safe Clone
                    return Pack(signal, 0f, true, triggerSnapshot);
                }
                return Pack(signal, reportingScore, false, null);
            }

            // 3. MASTER GATING
            bool allowed = DetectionEnabled && gateOpen;
            if (!allowed)
            {
                _state = RippleState.NoRipple;
                _scoreTicks = 0;
                _ticksInPossible = 0;
                return Pack(signal, 0f, false, null);
            }

            // 4. ACCUMULATOR FSM
            float scoreTarget = EventScoreThreshold * 2.0f;

            switch (_state)
            {
                case RippleState.NoRipple:
                    if (signal >= GateThreshold)
                    {
                        _state = RippleState.Possible;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;
                    }
                    reportingScore = 0f;
                    break;

                case RippleState.Possible:
                    _ticksInPossible++;

                    if (signal >= ConfirmThreshold) _scoreTicks += 2.0f;
                    else if (signal >= EnterThreshold) _scoreTicks += 1.0f;

                    reportingScore = _scoreTicks;

                    if (_scoreTicks >= scoreTarget)
                    {
                        _state = RippleState.Ripple;
                        _eventCount++;

                        float peakScore = _scoreTicks;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;

                        if (TriggerDelayMs > 0)
                        {
                            _ttlArmed = true;
                            _ttlAtMs = now + TriggerDelayMs;
                            return Pack(signal, peakScore, false, null);
                        }
                        else
                        {
                            _ttlHolding = true;
                            _ttlHoldUntilMs = now + Math.Max(1, PostRippleMs);
                            triggerSnapshot = rawInput?.Clone(); // Safe Clone
                            return Pack(signal, peakScore, true, triggerSnapshot);
                        }
                    }
                    else if (signal < GateThreshold)
                    {
                        _state = RippleState.NoRipple;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;
                        reportingScore = 0f;
                    }
                    else if (_ticksInPossible > GraceSamples)
                    {
                        _scoreTicks -= GraceRate;
                        if (_scoreTicks < 0) _scoreTicks = 0;
                        reportingScore = _scoreTicks;
                    }
                    break;

                case RippleState.Ripple:
                    if (signal < GateThreshold)
                    {
                        _state = RippleState.NoRipple;
                        _scoreTicks = 0;
                        reportingScore = 0f;
                    }
                    break;
            }

            return Pack(signal, reportingScore, _ttlHolding, triggerSnapshot);
        }

        private RippleOut Pack(float signal, float currentScore, bool ttl, Mat data)
        {
            return new RippleOut
            {
                State = _state,
                Score = currentScore * 0.5f,
                Probability = signal,
                EventCount = _eventCount,
                TTL = ttl,
                SignalData = data,
                StrideUsed = 0
            };
        }

        // ==============================================================================
        // OVERLOADS
        // ==============================================================================

        // Helper
        private float GetVal(Mat m)
        {
            if (m == null) return 0f;
            unsafe { return *((float*)m.Data.ToPointer()); }
        }

        // 1. Mat Only (Gate=True, No Data)
        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            return source.Select(m => Update(GetVal(m), true, null));
        }

        // 2. Mat + Bool (Prob, Gate) -> (No Data)
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
        {
            return source.Select(t => Update(GetVal(t.Item1), t.Item2, null));
        }

        // 3. ((Mat, Mat), Bool) -> ((Prob, RawData), Gate)
        // Wire: Zip(Prob, Raw) -> WithLatestFrom(Gate)
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
        {
            return source.Select(t =>
            {
                var mats = t.Item1;
                Mat prob = mats.Item1;
                Mat raw = mats.Item2;
                bool gate = t.Item2;

                return Update(GetVal(prob), gate, raw);
            });
        }
    }
}