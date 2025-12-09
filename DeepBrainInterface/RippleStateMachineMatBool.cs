using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    public enum RippleState { NoRipple, Possible, Ripple }

    // STRUCT: Created on Stack (Zero Garbage Collection pressure)
    public struct RippleOut
    {
        public RippleState State;
        public float Probability;
        public float ArtifactProbability;
        public int StrideUsed;
        public int EventCount;
        public float Score;
        public float LastEventScore;

        public bool EventPulse;
        public bool TriggerPulse;
        public bool TTL;

        public Mat TriggerData; // Class (Reference), this specific field CAN be null.
    }

    [Combinator]
    [Description("Logic Engine: Probability Mat -> RippleState.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class RippleStateMachineMatBool
    {
        // --- 1. PARAMETERS ---
        [Category("General")] public bool DetectionEnabled { get; set; } = true;
        [Category("General")] public float ArtifactThreshold { get; set; } = 0.5f;

        [Category("Thresholds")] public float GateThreshold { get; set; } = 0.10f;
        [Category("Thresholds")] public float EnterThreshold { get; set; } = 0.50f;
        [Category("Thresholds")] public float ConfirmThreshold { get; set; } = 0.80f;
        [Category("Thresholds")] public float EventScoreThreshold { get; set; } = 2.5f;
        [Category("Thresholds")] public float DecayRate { get; set; } = 1.0f;
        [Category("Thresholds")] public int DecayGraceTicks { get; set; } = 5;

        [Category("TTL")] public int TriggerDelayMs { get; set; } = 0;
        [Category("TTL")] public int PostRippleMs { get; set; } = 50;

        // --- 2. STATE ---
        RippleState _state = RippleState.NoRipple;
        float _scoreTicks;
        int _eventCount;
        float _lastEventScore;
        int _ticksInPossible;

        bool _ttlArmed; long _ttlAtMs;
        bool _ttlHolding; long _ttlHoldUntilMs;
        static readonly Stopwatch Clock = Stopwatch.StartNew();

        // --- 3. LOGIC ---
        public RippleOut Update(float signal, float artifact, bool bnoOk, Mat rawInput)
        {
            long now = Clock.ElapsedMilliseconds;
            bool triggerFrame = false;
            Mat triggerSnapshot = null;

            // A. HOLD PHASE
            if (_ttlHolding)
            {
                if (now >= _ttlHoldUntilMs)
                {
                    _ttlHolding = false;
                    _state = RippleState.NoRipple;
                    _scoreTicks = 0;
                    _ticksInPossible = 0;
                }
                else return Pack(signal, artifact, false, true, null);
            }

            // B. DELAY PHASE
            if (_ttlArmed)
            {
                if (now >= _ttlAtMs)
                {
                    _ttlArmed = false;
                    _ttlHolding = true;
                    _ttlHoldUntilMs = now + Math.Max(1, PostRippleMs);
                    triggerFrame = true;
                    // Only Allocate memory on trigger event
                    triggerSnapshot = rawInput?.Clone();
                    return Pack(signal, artifact, true, true, triggerSnapshot);
                }
                return Pack(signal, artifact, false, false, null);
            }

            // C. GATING
            bool artifactOk = artifact < ArtifactThreshold;
            if (!DetectionEnabled || !bnoOk || !artifactOk)
            {
                _state = RippleState.NoRipple; _scoreTicks = 0; _ticksInPossible = 0;
                return Pack(signal, artifact, false, false, null);
            }

            // D. FSM
            float eventTicksTarget = EventScoreThreshold * 2.0f;
            switch (_state)
            {
                case RippleState.NoRipple:
                    if (signal >= GateThreshold) { _state = RippleState.Possible; _scoreTicks = 0; _ticksInPossible = 0; }
                    break;

                case RippleState.Possible:
                    _ticksInPossible++;
                    if (signal >= ConfirmThreshold) _scoreTicks += 2.0f;
                    else if (signal >= EnterThreshold) _scoreTicks += 1.0f;

                    if (_scoreTicks >= eventTicksTarget)
                    {
                        _state = RippleState.Ripple;
                        _eventCount++;
                        _lastEventScore = _scoreTicks * 0.5f;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;

                        if (TriggerDelayMs > 0)
                        {
                            _ttlArmed = true; _ttlAtMs = now + TriggerDelayMs;
                            return Pack(signal, artifact, false, false, null);
                        }
                        else
                        {
                            triggerFrame = true;
                            triggerSnapshot = rawInput?.Clone();
                            _ttlHolding = true;
                            _ttlHoldUntilMs = now + Math.Max(1, PostRippleMs);
                        }
                    }
                    else if (signal < GateThreshold)
                    {
                        _state = RippleState.NoRipple; _scoreTicks = 0; _ticksInPossible = 0;
                    }
                    else if (_ticksInPossible > DecayGraceTicks)
                    {
                        _scoreTicks -= DecayRate;
                        if (_scoreTicks < 0) _scoreTicks = 0;
                    }
                    break;

                case RippleState.Ripple:
                    if (signal < GateThreshold) { _state = RippleState.NoRipple; _scoreTicks = 0; _ticksInPossible = 0; }
                    break;
            }

            return Pack(signal, artifact, triggerFrame, _ttlHolding, triggerSnapshot);
        }

        private RippleOut Pack(float signal, float artifact, bool pulse, bool ttl, Mat data)
        {
            // IMPORTANT: Returns a STRUCT. Never returns null.
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
                TriggerData = data // This field can be null, that's allowed.
            };
        }
    }
}