namespace DeepBrainInterface.Core
{
    /// <summary>Post-processing: turns the per-model probabilities of one inference into a decision.
    /// Symmetric to IPreprocessStage; RippleFsm is the default implementation.</summary>
    public interface IDecisionStage
    {
        RippleOut Update(float[] probs, bool gate, ulong clock);
        void Reset();
        int ConfigVersion { get; }
        int EventCount { get; }
    }
}
