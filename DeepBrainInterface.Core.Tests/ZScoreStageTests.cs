using System;
using DeepBrainInterface.Core;
using Xunit;

public class ZScoreStageTests
{
    [Fact]
    public void FirstSampleIsZero()
    {
        var s = new ZScoreStage(1, 4);
        var x = new float[] { 5.0f };
        s.Update(x);
        Assert.Equal(0f, x[0]);
    }

    [Fact]
    public void MatchesNaiveSlidingWindow()
    {
        const int ch = 3, win = 8, n = 50; const float eps = 1e-8f;
        var rnd = new Random(7);
        var series = new float[n, ch];
        for (int i = 0; i < n; i++) for (int c = 0; c < ch; c++) series[i, c] = (float)(rnd.NextDouble() * 200 - 100);
        var s = new ZScoreStage(ch, win, eps);
        var x = new float[ch];
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < ch; c++) x[c] = series[i, c];
            s.Update(x);
            int L = Math.Min(i + 1, win);
            for (int c = 0; c < ch; c++)
            {
                double sum = 0, sum2 = 0;
                for (int k = i - L + 1; k <= i; k++) { double v = series[k, c]; sum += v; sum2 += v * v; }
                double mu = sum / L; double var = sum2 / L - mu * mu; if (var < 0) var = 0;
                float naive = (float)((series[i, c] - mu) / (Math.Sqrt(var) + eps));
                Assert.True(Math.Abs(naive - x[c]) <= 1e-6 * Math.Max(1.0, Math.Abs(naive)), $"sample {i} ch {c}: {x[c]} vs {naive}");
            }
        }
    }

    [Fact]
    public void ResetRestartsFromEmpty()
    {
        var s = new ZScoreStage(2, 4);
        var x = new float[] { 1f, 2f };
        s.Update(x); x[0] = 3f; x[1] = 4f; s.Update(x);
        s.Reset();
        x[0] = 9f; x[1] = -9f; s.Update(x);
        Assert.Equal(0f, x[0]); Assert.Equal(0f, x[1]);
    }
}
