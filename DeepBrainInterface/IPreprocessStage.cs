namespace DeepBrainInterface.Core
{
    /// <summary>One preprocessing step applied in place to the channel vector of one (decimated) sample.
    /// Stages are chained by the frame processor; each phase owns its own instances. No allocation in Update.</summary>
    public interface IPreprocessStage
    {
        int Channels { get; }
        void Update(float[] x);
        void Reset();
    }
}
