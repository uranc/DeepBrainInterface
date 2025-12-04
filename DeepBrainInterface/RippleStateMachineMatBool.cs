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

        public bool EventPulse { get; set; }
        public bool TriggerPulse { get; set; }
        public bool TTL { get; set; }

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
        [Description("Score subtraction per tick when in Possible state (weak signal).")]
        public float DecayRate { get; set; } = 1.0f;

        [Category("Thresholds"), DisplayName("6. Decay Grace (ticks)")]
        public int DecayGraceTicks { get; set; } = 5;

        // ---- Timing ----
        [Category("TTL"), DisplayName("Trigger Delay (ms)")]
        public int TriggerDelayMs { get; set; } = 0;

        [Category("TTL"), DisplayName("PostRipple Hold (ms)")]
        public int PostRippleMs { get; set; } = 50;

        // ---- Internal ----
        RippleState _state = RippleState.NoRipple;
        float _scoreTicks;
        int _eventCount;
        float _lastEventScore;
        int _ticksInPossible;

        bool _ttlArmed;
        long _ttlAtMs;
        bool _ttlHolding;
        long _ttlHoldUntilMs;

        static readonly Stopwatch Clock = Stopwatch.StartNew();

        public RippleOut Update(float signal, float artifact, bool bnoOk, Mat rawInput)
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
                    _ticksInPossible = 0;
                }
                else
                {
                    return Pack(signal, artifact, false, true, null);
                }
            }

            // B. Delayed Trigger
            if (_ttlArmed && now >= _ttlAtMs)
            {
                _ttlArmed = false;
                _ttlHolding = true;
                _ttlHoldUntilMs = now + Math.Max(0, PostRippleMs);
                trigger = true;
                triggerSnapshot = rawInput?.Clone();
                return Pack(signal, artifact, true, true, triggerSnapshot);
            }

            // C. Gating Logic
            bool artifactOk = artifact < ArtifactThreshold;
            bool gatesOn = DetectionEnabled && bnoOk && artifactOk;

            if (!gatesOn)
            {
                _state = RippleState.NoRipple;
                _scoreTicks = 0;
                _ticksInPossible = 0;
                return Pack(signal, artifact, false, false, null);
            }

            // D. FSM
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

                    if (signal >= ConfirmThreshold)
                    {
                        // Strong signal
                        if (signal >= ConfirmThreshold) _scoreTicks += 2.0f;
                        else if (signal >= EnterThreshold) _scoreTicks += 1.0f;

                        // Trigger Check
                        if (_scoreTicks >= eventTicksTarget)
                        {
                            _state = RippleState.Ripple;
                            _eventCount++;
                            _lastEventScore = _scoreTicks * 0.5f;
                            _scoreTicks = 0;
                            _ticksInPossible = 0;

                            if (TriggerDelayMs > 0)
                            {
                                _ttlArmed = true;
                                _ttlAtMs = now + TriggerDelayMs;
                            }
                            else
                            {
                                trigger = true;
                                triggerSnapshot = rawInput?.Clone();
                                _ttlHolding = true;
                                _ttlHoldUntilMs = now + Math.Max(0, PostRippleMs);
                            }
                        }
                    }
                    else if (signal < GateThreshold)
                    {
                        // Drop below Gate -> Immediate Reset
                        _state = RippleState.NoRipple;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;
                    }
                    else
                    {
                        // Weak signal (Gate <= signal < Confirm)
                        // Apply Decay after grace period
                        if (_ticksInPossible > DecayGraceTicks)
                        {
                            _scoreTicks -= DecayRate;
                            if (_scoreTicks < 0) _scoreTicks = 0; // Clamp at 0
                        }
                    }
                    break;

                case RippleState.Ripple:
                    // Exit Logic
                    if (signal < GateThreshold)
                    {
                        _state = RippleState.NoRipple;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;
                    }
                    break;
            }

            return Pack(signal, artifact, trigger, _ttlHolding || _ttlArmed, triggerSnapshot);
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

        // Compatibility Overloads
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