using System;
using DeepBrainInterface.Core;
using Xunit;

public class StageTests
{
    [Fact]
    public void EmaConvergesToUnitVariance()
    {
        var s = new EmaStage(1, 0.01f, 1e-8f);
        var rnd = new Random(3); var x = new float[1];
        double sum = 0, sum2 = 0; int n = 0;
        for (int i = 0; i < 20000; i++)
        {
            double u1 = 1 - rnd.NextDouble(), u2 = rnd.NextDouble();
            x[0] = (float)(5 + 3 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
            s.Update(x);
            if (i >= 10000) { sum += x[0]; sum2 += x[0] * x[0]; n++; }
        }
        double mean = sum / n, std = Math.Sqrt(sum2 / n - mean * mean);
        Assert.InRange(mean, -0.1, 0.1); Assert.InRange(std, 0.9, 1.1);
    }

    [Fact]
    public void CommonAverageRemovesOffset()
    {
        var s = ReReferenceStage.CommonAverage(4);
        var x = new float[] { 11f, 12f, 13f, 14f };
        s.Update(x);
        Assert.InRange(x[0] + x[1] + x[2] + x[3], -1e-5, 1e-5);
        Assert.InRange(x[3] - x[0], 3 - 1e-5, 3 + 1e-5);
    }

    [Fact]
    public void CsdOfLinearRampIsZero()
    {
        var s = new CsdStage(5);
        var x = new float[] { 1f, 2f, 3f, 4f, 5f };
        s.Update(x);
        Assert.Equal(0f, x[1]); Assert.Equal(0f, x[2]); Assert.Equal(0f, x[3]);
        Assert.Equal(1f, x[0]); Assert.Equal(5f, x[4]);   // edges copied
    }

    [Fact]
    public void HighpassRemovesDc()
    {
        // 2nd-order Butterworth highpass, cutoff 0.01 of fs (scipy.signal.butter(2, 0.02, 'high') with Wn = 2*fc/fs)
        float[] b = { 0.95654323f, -1.91308646f, 0.95654323f };
        float[] a = { 1f, -1.91119707f, 0.91497583f };
        var s = new HighpassStage(2, b, a);
        var x = new float[2];
        for (int i = 0; i < 2000; i++) { x[0] = 100f; x[1] = -50f; s.Update(x); }
        Assert.InRange(Math.Abs(x[0]), 0, 1e-3); Assert.InRange(Math.Abs(x[1]), 0, 1e-3);
    }
}
