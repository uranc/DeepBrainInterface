using System;

namespace DeepBrainInterface.Core
{
    /// <summary>Exact rectangular sliding-window z-score, the math of the SlidingWindowZScore node:
    /// double sums, L = min(samples, window), var = max(E[x^2] - mu^2, 0), z = (x - mu) / (sqrt(var) + eps).</summary>
    public sealed class ZScoreStage : IPreprocessStage
    {
        readonly int channels, windowSize;
        readonly float eps;
        readonly double[] sumX, sumX2;
        readonly float[,] history;
        int historyIndex;
        long sampleCount;

        public ZScoreStage(int channels, int windowSize, float eps = 1e-8f)
        {
            if (channels < 1) throw new ArgumentOutOfRangeException(nameof(channels));
            if (windowSize < 1) throw new ArgumentOutOfRangeException(nameof(windowSize));
            this.channels = channels; this.windowSize = windowSize; this.eps = eps;
            sumX = new double[channels]; sumX2 = new double[channels]; history = new float[channels, windowSize];
        }

        public int Channels { get { return channels; } }

        public void Update(float[] x)
        {
            sampleCount++;
            double L = Math.Min(sampleCount, windowSize);
            for (int c = 0; c < channels; c++)
            {
                float val = x[c];
                double valD = val;
                if (sampleCount > windowSize)
                {
                    float oldVal = history[c, historyIndex];
                    sumX[c] -= oldVal;
                    sumX2[c] -= (double)oldVal * oldVal;
                }
                history[c, historyIndex] = val;
                sumX[c] += valD;
                sumX2[c] += valD * valD;
                double mu = sumX[c] / L;
                double var = (sumX2[c] / L) - (mu * mu);
                if (var < 0.0) var = 0.0;
                double sig = Math.Sqrt(var);
                x[c] = (float)((valD - mu) / (sig + eps));
            }
            historyIndex = (historyIndex + 1) % windowSize;
        }

        public void Reset()
        {
            Array.Clear(sumX, 0, channels); Array.Clear(sumX2, 0, channels); Array.Clear(history, 0, history.Length);
            historyIndex = 0; sampleCount = 0;
        }
    }
}
