using System;

namespace DeepBrainInterface.Core
{
    public enum Combiner { First, Max, Mean }

    /// <summary>Start-time configuration of a FrameProcessor. Everything here is fixed after construction;
    /// ActiveModels and Combiner also have live setters on the processor.</summary>
    public sealed class ProcessorConfig
    {
        /// <summary>0-based amplifier rows to read, in the order the model expects.</summary>
        public int[] Channels;
        /// <summary>Keep every DecimationFactor-th raw column (phase 0 = first sample). 1 = raw rate.</summary>
        public int DecimationFactor = 12;
        /// <summary>Number of decimation phases processed (1 = plain decimation; = DecimationFactor for all shifted copies).</summary>
        public int PhaseCount = 1;
        public int TimePoints = 44;
        /// <summary>Inference every Stride kept samples of phase 0.</summary>
        public int Stride = 2;
        public string[] ModelPaths;
        /// <summary>Preprocessing chain for phase p (each phase gets its own instances).</summary>
        public Func<int, IPreprocessStage[]> StageFactory;
        /// <summary>0 = model 0 only, 1 = model 1 only, 2 = both.</summary>
        public int ActiveModels = 0;
        public Combiner Combiner = Combiner.First;
        /// <summary>Attach the phase-0 model input to RippleOut on a TTL rise (reused array).</summary>
        public bool SnapshotEnabled = false;
    }
}
