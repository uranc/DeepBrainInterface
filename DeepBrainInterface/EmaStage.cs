using System;

namespace DeepBrainInterface.Core
{
    /// <summary>Per-channel exponential mean/variance normalisation: m += a(x-m); v += a((x-m)^2 - v); z = (x-m)/(sqrt(v)+eps).</summary>
    public sealed class EmaStage : IPreprocessStage
    {
        readonly int channels; readonly float alpha, eps;
        readonly double[] mean, var;
        bool primed;

        public EmaStage(int channels, float alpha, float eps = 1e-8f)
        {
            if (channels < 1) throw new ArgumentOutOfRangeException(nameof(channels));
            if (alpha <= 0 || alpha > 1) throw new ArgumentOutOfRangeException(nameof(alpha));
            this.channels = channels; this.alpha = alpha; this.eps = eps;
            mean = new double[channels]; var = new double[channels];
        }

        public int Channels { get { return channels; } }

        public void Update(float[] x)
        {
            if (!primed) { for (int c = 0; c < channels; c++) { mean[c] = x[c]; var[c] = 0; x[c] = 0f; } primed = true; return; }
            for (int c = 0; c < channels; c++)
            {
                double d = x[c] - mean[c];
                mean[c] += alpha * d;
                var[c] += alpha * (d * d - var[c]);
                x[c] = (float)((x[c] - mean[c]) / (Math.Sqrt(var[c]) + eps));
            }
        }

        public void Reset() { Array.Clear(mean, 0, channels); Array.Clear(var, 0, channels); primed = false; }
    }
}
