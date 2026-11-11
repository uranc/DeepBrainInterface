using System;

namespace DeepBrainInterface.Core
{
    /// <summary>Spatial re-reference y = W x with a fixed channels x channels weight matrix; CommonAverage(n) gives I - 1/n.</summary>
    public sealed class ReReferenceStage : IPreprocessStage
    {
        readonly int channels; readonly float[,] w; readonly float[] tmp;

        public ReReferenceStage(int channels, float[,] weights)
        {
            if (weights.GetLength(0) != channels || weights.GetLength(1) != channels) throw new ArgumentException("weights must be channels x channels", nameof(weights));
            this.channels = channels; w = weights; tmp = new float[channels];
        }

        public static ReReferenceStage CommonAverage(int channels)
        {
            var w = new float[channels, channels]; float k = 1f / channels;
            for (int i = 0; i < channels; i++) for (int j = 0; j < channels; j++) w[i, j] = (i == j ? 1f : 0f) - k;
            return new ReReferenceStage(channels, w);
        }

        public int Channels { get { return channels; } }

        public void Update(float[] x)
        {
            for (int i = 0; i < channels; i++) { double acc = 0; for (int j = 0; j < channels; j++) acc += w[i, j] * x[j]; tmp[i] = (float)acc; }
            Array.Copy(tmp, x, channels);
        }

        public void Reset() { }
    }
}
