using System;
using System.Diagnostics;

namespace DeepBrainInterface.Core
{
    /// <summary>Minimal decision stage: TTL after N consecutive decisions with probs[0] >= threshold, then hold PostRippleMs.</summary>
    public sealed class ThresholdHoldStage : IDecisionStage
    {
        public float Threshold; public int Consecutive; public int PostRippleMs;
        public Func<long> NowMs;
        public int ConfigVersion { get; set; }
        static readonly Stopwatch Wall = Stopwatch.StartNew();
        int run, eventCount; bool holding; long holdUntil;

        public ThresholdHoldStage(float threshold, int consecutive, int postRippleMs, Func<long> nowMs = null)
        {
            Threshold = threshold; Consecutive = Math.Max(1, consecutive); PostRippleMs = postRippleMs; NowMs = nowMs ?? (() => Wall.ElapsedMilliseconds);
        }

        public int EventCount { get { return eventCount; } }
        public void Reset() { run = 0; holding = false; holdUntil = 0; }

        public RippleOut Update(float[] probs, bool gate, ulong clock)
        {
            long now = NowMs(); float p = probs[0]; bool ttl = false;
            if (holding) { if (now >= holdUntil) { holding = false; run = 0; } else ttl = true; }
            if (!holding)
            {
                if (gate && p >= Threshold) run++; else run = 0;
                if (run >= Consecutive) { eventCount++; run = 0; holding = true; holdUntil = now + Math.Max(1, PostRippleMs); ttl = true; }
            }
            return new RippleOut
            {
                State = holding ? RippleState.Ripple : (run > 0 ? RippleState.Possible : RippleState.NoRipple), TTL = ttl, Score = run,
                Probability = p, EventCount = eventCount, Clock = clock, Threshold1 = Threshold, EvidenceTarget = Consecutive, PostRippleMs = PostRippleMs,
                DetectionEnabled = true, ConfigVersion = ConfigVersion, Prob0 = p, Prob1 = probs.Length > 1 ? probs[1] : float.NaN, TriggerDelaySetting = "0",
            };
        }
    }
}
