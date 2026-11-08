using System;
using System.Diagnostics;

namespace DeepBrainInterface.Core
{
    /// <summary>Accumulator FSM with leaky-bucket grace period: the state machine of RippleStateMachineMatBool (1.1.76), verbatim,
    /// with the wall clock injected (NowMs) so it is testable. Uses probs[0]. No allocation in Update.</summary>
    public sealed class RippleFsm : IDecisionStage
    {
        public bool DetectionEnabled = true;
        public float Threshold1 = 0.10f, Threshold2 = 0.50f, Threshold3 = 0.80f;
        public float TargetEvidenceScore = 5.0f;
        public int FramesToWaitBeforeDecay = 5;
        public float ScoreDecayPerFrame = 1.0f;
        public int DelayMinMs, DelayMaxMs;
        public int PostRippleMs = 50;
        public string TriggerDelaySetting = "0";
        public Func<long> NowMs;
        /// <summary>When > 0, hold and delay timing use the hardware clock of each decision (ms = clock * 1000 / Hz) instead of the wall clock:
        /// deterministic, and required for offline replay.</summary>
        public double HardwareClockHz;
        long hwNowMs;
        public int ConfigVersion { get; set; }

        static readonly Stopwatch Wall = Stopwatch.StartNew();
        readonly Random random = new Random();

        RippleState state = RippleState.NoRipple;
        float scoreTicks; int ticksDipping; int eventCount;
        bool ttlArmed; long ttlAtMs; bool ttlHolding; long ttlHoldUntilMs; int currentEventDelay;

        public RippleFsm() { NowMs = () => Wall.ElapsedMilliseconds; }

        public int EventCount { get { return eventCount; } }

        public void Reset()
        {
            state = RippleState.NoRipple; scoreTicks = 0; ticksDipping = 0;
            ttlArmed = false; ttlAtMs = 0; ttlHolding = false; ttlHoldUntilMs = 0; currentEventDelay = DelayMinMs;
        }

        public RippleOut Update(float[] probs, bool gate, ulong clock)
        {
            if (HardwareClockHz > 0) hwNowMs = (long)(clock * 1000.0 / HardwareClockHz);
            var r = Update(probs[0], 0f, gate);
            r.Clock = clock; r.Prob0 = probs[0]; r.Prob1 = probs.Length > 1 ? probs[1] : float.NaN;
            return r;
        }

        public RippleOut Update(float signal, float artProb, bool gateOpen)
        {
            long now = HardwareClockHz > 0 ? hwNowMs : NowMs();
            float reportingScore = scoreTicks;

            // 1. TTL refractory period
            if (ttlHolding)
            {
                if (now >= ttlHoldUntilMs)
                {
                    ttlHolding = false; state = RippleState.NoRipple; scoreTicks = 0; ticksDipping = 0; reportingScore = 0f;
                }
                else return Pack(signal, artProb, 0f, true);
            }

            // 2. Delayed TTL trigger
            if (ttlArmed)
            {
                if (now >= ttlAtMs)
                {
                    ttlArmed = false; ttlHolding = true; ttlHoldUntilMs = now + Math.Max(1, PostRippleMs);
                    return Pack(signal, artProb, 0f, true);
                }
                return Pack(signal, artProb, reportingScore, false);
            }

            // 3. Master gate
            bool allowed = DetectionEnabled && gateOpen;
            if (!allowed)
            {
                state = RippleState.NoRipple; scoreTicks = 0; ticksDipping = 0;
                return Pack(signal, artProb, 0f, false);
            }

            // 4. State machine
            switch (state)
            {
                case RippleState.NoRipple:
                    if (signal >= Threshold1) { state = RippleState.Possible; scoreTicks = 0; ticksDipping = 0; }
                    reportingScore = 0f;
                    break;

                case RippleState.Possible:
                    if (signal >= Threshold1)
                    {
                        ticksDipping = 0;
                        if (signal >= Threshold3) scoreTicks += 2.5f;
                        else if (signal >= Threshold2) scoreTicks += 1.0f;
                        reportingScore = scoreTicks;
                        if (scoreTicks >= TargetEvidenceScore)
                        {
                            state = RippleState.Ripple; eventCount++;
                            float peakScore = scoreTicks; scoreTicks = 0; ticksDipping = 0;
                            if (DelayMinMs > 0 || DelayMaxMs > 0)
                            {
                                ttlArmed = true;
                                currentEventDelay = (DelayMinMs >= DelayMaxMs) ? DelayMinMs : random.Next(DelayMinMs, DelayMaxMs + 1);
                                ttlAtMs = now + currentEventDelay;
                                return Pack(signal, artProb, peakScore, false);
                            }
                            currentEventDelay = 0; ttlHolding = true; ttlHoldUntilMs = now + Math.Max(1, PostRippleMs);
                            return Pack(signal, artProb, peakScore, true);
                        }
                    }
                    else
                    {
                        ticksDipping++;
                        if (ticksDipping > FramesToWaitBeforeDecay)
                        {
                            scoreTicks -= ScoreDecayPerFrame;
                            if (scoreTicks <= 0) { state = RippleState.NoRipple; scoreTicks = 0; ticksDipping = 0; reportingScore = 0f; }
                            else reportingScore = scoreTicks;
                        }
                        else reportingScore = scoreTicks;
                    }
                    break;

                case RippleState.Ripple:
                    if (signal < Threshold1) { state = RippleState.NoRipple; scoreTicks = 0; ticksDipping = 0; reportingScore = 0f; }
                    break;
            }
            return Pack(signal, artProb, reportingScore, ttlHolding);
        }

        RippleOut Pack(float signal, float artProb, float currentScore, bool ttl)
        {
            return new RippleOut
            {
                State = state, Score = currentScore, Probability = signal, ArtifactProbability = artProb, EventCount = eventCount, TTL = ttl, Skipped = 0,
                Threshold1 = Threshold1, Threshold2 = Threshold2, Threshold3 = Threshold3, EvidenceTarget = TargetEvidenceScore,
                PostRippleMs = PostRippleMs, TriggerDelayMs = currentEventDelay, DetectionEnabled = DetectionEnabled,
                FramesToWaitBeforeDecay = FramesToWaitBeforeDecay, ScoreDecayPerFrame = ScoreDecayPerFrame, TriggerDelaySetting = TriggerDelaySetting,
                ConfigVersion = ConfigVersion, Prob0 = float.NaN, Prob1 = float.NaN,
            };
        }
    }
}
