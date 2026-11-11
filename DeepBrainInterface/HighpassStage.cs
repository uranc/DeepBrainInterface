using System;

namespace DeepBrainInterface.Core
{
    /// <summary>Per-channel IIR filter (direct form II transposed) with coefficients b[0..N], a[0..N] (a[0] = 1).</summary>
    public sealed class HighpassStage : IPreprocessStage
    {
        readonly int channels, order; readonly double[] b, a; readonly double[,] z;

        public HighpassStage(int channels, float[] b, float[] a)
        {
            if (b.Length != a.Length || b.Length < 2) throw new ArgumentException("b and a must have the same length >= 2");
            if (Math.Abs(a[0] - 1f) > 1e-6f) throw new ArgumentException("a[0] must be 1", nameof(a));
            this.channels = channels; order = b.Length - 1;
            this.b = new double[b.Length]; this.a = new double[a.Length];
            for (int i = 0; i < b.Length; i++) { this.b[i] = b[i]; this.a[i] = a[i]; }
            z = new double[channels, order];
        }

        public int Channels { get { return channels; } }

        public void Update(float[] x)
        {
            for (int c = 0; c < channels; c++)
            {
                double xin = x[c];
                double y = b[0] * xin + z[c, 0];
                for (int k = 0; k < order - 1; k++) z[c, k] = b[k + 1] * xin - a[k + 1] * y + z[c, k + 1];
                z[c, order - 1] = b[order] * xin - a[order] * y;
                x[c] = (float)y;
            }
        }

        public void Reset() { Array.Clear(z, 0, z.Length); }
    }
}
