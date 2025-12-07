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

        // Outputs
        public bool EventPulse { get; set; }   // Single frame High on detection
        public bool TriggerPulse { get; set; } // Same as EventPulse
        public bool TTL { get; set; }          // High during Hold Duration

        public Mat TriggerData { get; set; }   // Snapshot of the input (if provided)
    }

    [Combinator]
    [Description("Logic Engine: Probability Mat -> RippleState (with optional gating).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class RippleStateMachineMatBool
    {
        // ==============================================================================
        // 1. PARAMETERS
        // ==============================================================================
        [Category("General"), DisplayName("Detection Enabled")] public bool DetectionEnabled { get; set; } = true;
        [Category("General"), DisplayName("Artifact Threshold")] public float ArtifactThreshold { get; set; } = 0.5f;

        [Category("Thresholds"), DisplayName("1. Gate (arm)")] public float GateThreshold { get; set; } = 0.10f;
        [Category("Thresholds"), DisplayName("2. Enter (+0.5)")] public float EnterThreshold { get; set; } = 0.50f;
        [Category("Thresholds"), DisplayName("3. Confirm (+1.0)")] public float ConfirmThreshold { get; set; } = 0.80f;
        [Category("Thresholds"), DisplayName("4. Event Score")] public float EventScoreThreshold { get; set; } = 2.5f;
        [Category("Thresholds"), DisplayName("5. Decay Rate")] public float DecayRate { get; set; } = 1.0f;
        [Category("Thresholds"), DisplayName("6. Decay Grace")] public int DecayGraceTicks { get; set; } = 5;

        [Category("TTL"), DisplayName("Trigger Delay (ms)")] public int TriggerDelayMs { get; set; } = 0;
        [Category("TTL"), DisplayName("PostRipple Hold (ms)")] public int PostRippleMs { get; set; } = 50;

        // ==============================================================================
        // 2. INTERNAL STATE
        // ==============================================================================
        RippleState _state = RippleState.NoRipple;
        float _scoreTicks;
        int _eventCount;
        float _lastEventScore;
        int _ticksInPossible;

        // Trigger Logic
        bool _ttlArmed; long _ttlAtMs;
        bool _ttlHolding; long _ttlHoldUntilMs;
        static readonly Stopwatch Clock = Stopwatch.StartNew();

        // ==============================================================================
        // 3. CORE LOGIC
        // ==============================================================================
        public RippleOut Update(float signal, float artifact, bool bnoOk, Mat rawInput)
        {
            long now = Clock.ElapsedMilliseconds;
            bool triggerFrame = false;
            Mat triggerSnapshot = null;

            // A. HOLD PHASE (Refractory)
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

            // B. DELAY PHASE (Latched Wait)
            if (_ttlArmed)
            {
                if (now >= _ttlAtMs)
                {
                    // Delay Over -> FIRE
                    _ttlArmed = false;
                    _ttlHolding = true;
                    _ttlHoldUntilMs = now + Math.Max(1, PostRippleMs);

                    triggerFrame = true;
                    triggerSnapshot = rawInput?.Clone();

                    return Pack(signal, artifact, true, true, triggerSnapshot);
                }
                // Waiting... (TTL Low)
                return Pack(signal, artifact, false, false, null);
            }

            // C. GATING (Input Validation)
            bool artifactOk = artifact < ArtifactThreshold;
            bool gatesOn = DetectionEnabled && bnoOk && artifactOk;

            if (!gatesOn)
            {
                _state = RippleState.NoRipple; _scoreTicks = 0; _ticksInPossible = 0;
                return Pack(signal, artifact, false, false, null);
            }

            // D. FINITE STATE MACHINE
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
                            _ttlArmed = true; _ttlAtMs = now + TriggerDelayMs;
                            return Pack(signal, artifact, false, false, null);
                        }
                        else
                        {
                            triggerFrame = true; triggerSnapshot = rawInput?.Clone();
                            _ttlHolding = true; _ttlHoldUntilMs = now + Math.Max(1, PostRippleMs);
                        }
                    }
                    // Drop Out
                    else if (signal < GateThreshold)
                    {
                        _state = RippleState.NoRipple; _scoreTicks = 0; _ticksInPossible = 0;
                    }
                    // Decay
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
                TriggerData = data
            };
        }

        // ==============================================================================
        // 4. OVERLOADS (The Critical Fix)
        // ==============================================================================

        // Helper: Extracts (Signal, Artifact) from a 2x1 Mat
        private void ExtractProbs(Mat m, out float sig, out float art)
        {
            sig = 0; art = 0;
            if (m != null && m.Depth == Depth.F32)
            {
                unsafe
                {
                    float* ptr = (float*)m.Data.ToPointer();
                    int len = m.Rows * m.Cols;
                    if (len >= 1) sig = ptr[0];
                    if (len >= 2) art = ptr[1];
                }
            }
        }

        // >>> PRIMARY OVERLOAD: Accepts pure Mat (Output of Detector) <<<
        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            return source.Select(m => {
                ExtractProbs(m, out float s, out float a);
                // Default: Gate is TRUE (Open), Raw Input is the Prob Mat itself
                return Update(s, a, true, m);
            });
        }

        // Optional: Mat + Bool Gate (e.g. from WithLatestFrom BNO)
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
        {
            return source.Select(t => {
                ExtractProbs(t.Item1, out float s, out float a);
                return Update(s, a, t.Item2, t.Item1);
            });
        }

        // Optional: Mat + Int Gate (e.g. Arduino TTL line)
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, int>> source)
        {
            return source.Select(t => {
                ExtractProbs(t.Item1, out float s, out float a);
                return Update(s, a, t.Item2 != 0, t.Item1);
            });
        }
    }
}