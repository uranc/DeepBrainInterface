using System;

namespace DeepBrainInterface.Core
{
    /// <summary>Second spatial difference along the channel order: y[c] = x[c-1] - 2x[c] + x[c+1]; edge channels are copied through.</summary>
    public sealed class CsdStage : IPreprocessStage
    {
        readonly int channels; readonly float[] tmp;
        public CsdStage(int channels) { if (channels < 3) throw new ArgumentOutOfRangeException(nameof(channels)); this.channels = channels; tmp = new float[channels]; }
        public int Channels { get { return channels; } }
        public void Update(float[] x)
        {
            tmp[0] = x[0]; tmp[channels - 1] = x[channels - 1];
            for (int c = 1; c < channels - 1; c++) tmp[c] = x[c - 1] - 2f * x[c] + x[c + 1];
            Array.Copy(tmp, x, channels);
        }
        public void Reset() { }
    }
}
