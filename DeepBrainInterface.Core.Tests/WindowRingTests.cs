using System;
using DeepBrainInterface.Core;
using Xunit;

public class WindowRingTests
{
    [Fact]
    public void WindowIsOldestFirst()
    {
        var r = new WindowRing(4, 1); var x = new float[1];
        for (int i = 1; i <= 50; i++) { x[0] = i; r.Push(x); }
        var dst = new float[4]; r.CopyWindow(dst, 0);
        Assert.Equal(new float[] { 47, 48, 49, 50 }, dst);
    }

    [Fact]
    public void NotFullBeforeTimePoints()
    {
        var r = new WindowRing(4, 2); var x = new float[2];
        for (int i = 0; i < 3; i++) { Assert.False(r.IsFull); r.Push(x); }
        Assert.False(r.IsFull); r.Push(x); Assert.True(r.IsFull);
        r.Reset(); Assert.False(r.IsFull);
    }

    [Fact]
    public void CopyNeverWraps()
    {
        const int T = 44, C = 8; var r = new WindowRing(T, C); var x = new float[C];
        var last = new float[T, C];
        for (int i = 0; i < 1000; i++)
        {
            for (int c = 0; c < C; c++) x[c] = i * 10 + c;
            r.Push(x);
            for (int c = 0; c < C; c++) last[i % T, c] = x[c];
            if (i >= T - 1 && i % 7 == 0)
            {
                var dst = new float[T * C + 5]; r.CopyWindow(dst, 5);
                for (int t = 0; t < T; t++) for (int c = 0; c < C; c++) Assert.Equal(last[(i - T + 1 + t) % T, c], dst[5 + t * C + c]);
            }
        }
    }
}
