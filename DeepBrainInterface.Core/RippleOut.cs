namespace DeepBrainInterface.Core
{
    public enum RippleState { NoRipple, Possible, Ripple }

    /// <summary>One detector decision. Fields up to TriggerDelaySetting are the 1.1.76 layout (CSV selectors keep working);
    /// the rest are new in 1.1.78 and opt-in.</summary>
    public struct RippleOut
    {
        // --- Detection state ---
        public RippleState State { get; set; }
        public bool TTL { get; set; }
        public float Score { get; set; }
        public int EventCount { get; set; }
        public float Probability { get; set; }
        public float ArtifactProbability { get; set; }

        // --- Timing ---
        public ulong Clock { get; set; }  // hardware clock of the freshest sample
        public int Skipped { get; set; }  // 0 = inference ran normally; 1 = skipped (e.g. during refractory)

        // --- FSM config snapshot ---
        public float Threshold1 { get; set; }
        public float Threshold2 { get; set; }
        public float Threshold3 { get; set; }
        public float EvidenceTarget { get; set; }
        public int PostRippleMs { get; set; }
        public int TriggerDelayMs { get; set; }
        public bool DetectionEnabled { get; set; }
        public int FramesToWaitBeforeDecay { get; set; }
        public int DecayWaitMs { get; set; }        // ms grace on the hardware clock (0 = frames)
        public float ScoreDecayPerFrame { get; set; }
        public string TriggerDelaySetting { get; set; }

        // --- 1.1.78 ---
        public int ConfigVersion { get; set; }      // increments on any live parameter change
        public float Prob0 { get; set; }            // per-model probabilities (NaN when the model did not run)
        public float Prob1 { get; set; }
        public long DecisionIndex { get; set; }     // inferences since subscribe
        public float[] SnapshotWindow { get; set; } // model input at a TTL rise when enabled, else null
    }
}
