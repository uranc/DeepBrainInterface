using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    // SHARED TYPES
    public enum RippleState { NoRipple, Possible, Ripple }

    public sealed class RippleOut
    {
        public RippleState State { get; set; }
        public float Probability { get; set; }
        public float ArtifactProbability { get; set; }
        public int StrideUsed { get; set; }
        public int EventCount { get; set; }

        public float Score { get; set; }
        public float LastEventScore { get; set; }

        public bool EventPulse { get; set; }   // True only for 1 frame when trigger fires
        public bool TriggerPulse { get; set; } // Same as EventPulse
        public bool TTL { get; set; }          // True during the "Hold" phase

        public Mat TriggerData { get; set; }
    }

    [Combinator]
    [Description("Logic Engine: Probability + Artifact + BNO Gate → RippleState.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class RippleStateMachineMatBool
    {
        // ---- General ----
        [Category("General"), DisplayName("Detection Enabled")]
        public bool DetectionEnabled { get; set; } = true;

        [Category("General"), DisplayName("Artifact Threshold")]
        public float ArtifactThreshold { get; set; } = 0.5f;

        // ---- Thresholds ----
        [Category("Thresholds"), DisplayName("1. Gate (arm)")]
        public float GateThreshold { get; set; } = 0.10f;

        [Category("Thresholds"), DisplayName("2. Enter (+0.5 per tick)")]
        public float EnterThreshold { get; set; } = 0.50f;

        [Category("Thresholds"), DisplayName("3. Confirm (+1.0 per tick)")]
        public float ConfirmThreshold { get; set; } = 0.80f;

        [Category("Thresholds"), DisplayName("4. Event Score (≥ triggers)")]
        public float EventScoreThreshold { get; set; } = 2.5f;

        [Category("Thresholds"), DisplayName("5. Decay Rate")]
        public float DecayRate { get; set; } = 1.0f;

        [Category("Thresholds"), DisplayName("6. Decay Grace (ticks)")]
        public int DecayGraceTicks { get; set; } = 5;

        // ---- Timing ----
        [Category("TTL"), DisplayName("Trigger Delay (ms)")]
        [Description("Wait time AFTER detection before setting TTL High.")]
        public int TriggerDelayMs { get; set; } = 0;

        [Category("TTL"), DisplayName("PostRipple Hold (ms)")]
        [Description("Duration TTL stays High after the delay finishes.")]
        public int PostRippleMs { get; set; } = 50;

        // ---- Internal ----
        RippleState _state = RippleState.NoRipple;
        float _scoreTicks;
        int _eventCount;
        float _lastEventScore;
        int _ticksInPossible;

        bool _ttlArmed;         // Waiting for the Delay
        long _ttlAtMs;          // Target time to fire
        bool _ttlHolding;       // TTL is High
        long _ttlHoldUntilMs;   // Target time to drop TTL

        static readonly Stopwatch Clock = Stopwatch.StartNew();

        public RippleOut Update(float signal, float artifact, bool bnoOk, Mat rawInput)
        {
            long now = Clock.ElapsedMilliseconds;
            bool triggerFrame = false;
            Mat triggerSnapshot = null;

            // ---------------------------------------------------------
            // 1. POST-RIPPLE HOLD (TTL = HIGH)
            // ---------------------------------------------------------
            // If we are holding, we block everything else.
            if (_ttlHolding)
            {
                if (now >= _ttlHoldUntilMs)
                {
                    // Hold finished. Reset.
                    _ttlHolding = false;
                    _state = RippleState.NoRipple;
                    _scoreTicks = 0;
                    _ticksInPossible = 0;
                }
                else
                {
                    // Still holding. TTL is TRUE.
                    return Pack(signal, artifact, false, true, null);
                }
            }

            // ---------------------------------------------------------
            // 2. TRIGGER DELAY (TTL = LOW)
            // ---------------------------------------------------------
            // If armed, we wait. We ignore signal drops (latched).
            if (_ttlArmed)
            {
                if (now >= _ttlAtMs)
                {
                    // Delay finished. FIRE.
                    _ttlArmed = false;

                    _ttlHolding = true;
                    _ttlHoldUntilMs = now + Math.Max(1, PostRippleMs); // Ensure non-zero hold

                    triggerFrame = true;
                    triggerSnapshot = rawInput?.Clone();

                    // Pulse = True, TTL = True
                    return Pack(signal, artifact, true, true, triggerSnapshot);
                }

                // Still waiting. TTL is FALSE.
                return Pack(signal, artifact, false, false, null);
            }

            // ---------------------------------------------------------
            // 3. GATING (Input Validation)
            // ---------------------------------------------------------
            bool artifactOk = artifact < ArtifactThreshold;
            bool gatesOn = DetectionEnabled && bnoOk && artifactOk;

            if (!gatesOn)
            {
                _state = RippleState.NoRipple;
                _scoreTicks = 0;
                _ticksInPossible = 0;
                return Pack(signal, artifact, false, false, null);
            }

            // ---------------------------------------------------------
            // 4. DETECTION FSM
            // ---------------------------------------------------------
            float eventTicksTarget = EventScoreThreshold * 2.0f;

            switch (_state)
            {
                case RippleState.NoRipple:
                    if (signal >= GateThreshold)
                    {
                        _state = RippleState.Possible;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;
                    }
                    break;

                case RippleState.Possible:
                    _ticksInPossible++;

                    // Accumulate
                    if (signal >= ConfirmThreshold) _scoreTicks += 2.0f;
                    else if (signal >= EnterThreshold) _scoreTicks += 1.0f;

                    // Check Threshold
                    if (_scoreTicks >= eventTicksTarget)
                    {
                        // --- DETECTED ---
                        _state = RippleState.Ripple;
                        _eventCount++;
                        _lastEventScore = _scoreTicks * 0.5f;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;

                        if (TriggerDelayMs > 0)
                        {
                            // Arm the delay. Return TTL=False for now.
                            _ttlArmed = true;
                            _ttlAtMs = now + TriggerDelayMs;
                            return Pack(signal, artifact, false, false, null);
                        }
                        else
                        {
                            // No delay? Fire immediately.
                            triggerFrame = true;
                            triggerSnapshot = rawInput?.Clone();

                            _ttlHolding = true;
                            _ttlHoldUntilMs = now + Math.Max(1, PostRippleMs);
                        }
                    }
                    // Drop out
                    else if (signal < GateThreshold)
                    {
                        _state = RippleState.NoRipple;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;
                    }
                    // Decay
                    else if (_ticksInPossible > DecayGraceTicks)
                    {
                        _scoreTicks -= DecayRate;
                        if (_scoreTicks < 0) _scoreTicks = 0;
                    }
                    break;

                case RippleState.Ripple:
                    // Only exit if signal drops (handled here if detection continues without triggering)
                    // Note: If we triggered, we are in Block 1 or 2, so we never reach here.
                    if (signal < GateThreshold)
                    {
                        _state = RippleState.NoRipple;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;
                    }
                    break;
            }

            return Pack(signal, artifact, triggerFrame, _ttlHolding, triggerSnapshot);
        }

        private RippleOut Pack(float signal, float artifact, bool pulse, bool ttl, Mat data)
        {
            return new RippleOut
            {
                State = _state,
                Score = _scoreTicks * 0.5f,
                Probability = signal,
                ArtifactProbability = artifact,
                EventCount = _eventCount,
                LastEventScore = _lastEventScore,
                EventPulse = pulse,
                TriggerPulse = pulse,
                TTL = ttl,
                TriggerData = data,
                StrideUsed = 0
            };
        }

        // ---- Boilerplate (Unchanged) ----
        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            return source.Select(m =>
            {
                float sig = 0, art = 0;
                int totalElements = m.Rows * m.Cols * m.Channels;
                if (totalElements >= 2 && m.Depth == Depth.F32)
                {
                    unsafe
                    {
                        float* ptr = (float*)m.Data.ToPointer();
                        sig = ptr[0];
                        art = ptr[1];
                    }
                }
                else if (m.Depth == Depth.F32)
                {
                    unsafe { sig = *((float*)m.Data.ToPointer()); }
                }
                return Update(sig, art, true, m);
            });
        }

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
        {
            return source.Select(t =>
            {
                float sig = 0, art = 0;
                Mat m = t.Item1;
                int totalElements = m.Rows * m.Cols * m.Channels;
                if (totalElements >= 2 && m.Depth == Depth.F32)
                {
                    unsafe
                    {
                        float* ptr = (float*)m.Data.ToPointer();
                        sig = ptr[0];
                        art = ptr[1];
                    }
                }
                else if (m.Depth == Depth.F32)
                {
                    unsafe { sig = *((float*)m.Data.ToPointer()); }
                }
                return Update(sig, art, t.Item2, m);
            });
        }
    }
}