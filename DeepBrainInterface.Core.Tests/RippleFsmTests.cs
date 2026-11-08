using System;
using DeepBrainInterface.Core;
using Xunit;

public class RippleFsmTests
{
    long now;
    RippleFsm Make(float target = 5f)
    {
        now = 1000;
        return new RippleFsm { Threshold1 = 0.1f, Threshold2 = 0.5f, Threshold3 = 0.8f, TargetEvidenceScore = target, FramesToWaitBeforeDecay = 5, ScoreDecayPerFrame = 1f, PostRippleMs = 50, DetectionEnabled = true, NowMs = () => now };
    }
    static float[] P(float p) => new[] { p };

    [Fact]
    public void TriggersAfterTargetScore()
    {
        var f = Make(5f);
        Assert.False(f.Update(P(0.9f), true, 1).TTL);          // NoRipple -> Possible (no points yet, as in the node)
        Assert.False(f.Update(P(0.9f), true, 2).TTL);          // +2.5
        var r = f.Update(P(0.9f), true, 3);                    // +2.5 = 5 -> trigger
        Assert.True(r.TTL); Assert.Equal(1, r.EventCount); Assert.Equal(5f, r.Score); Assert.Equal(3UL, r.Clock);
    }

    [Fact]
    public void WeakAndStrongPoints()
    {
        var f = Make(100f);
        f.Update(P(0.9f), true, 1);
        Assert.Equal(1f, f.Update(P(0.6f), true, 2).Score);
        Assert.Equal(3.5f, f.Update(P(0.9f), true, 3).Score);
    }

    [Fact]
    public void LeakyBucketGrace()
    {
        var f = Make(100f);
        f.Update(P(0.9f), true, 1); f.Update(P(0.9f), true, 2);          // score 2.5
        for (int i = 0; i < 5; i++) Assert.Equal(2.5f, f.Update(P(0.05f), true, 10 + (ulong)i).Score);   // grace: 5 dips keep the score
        Assert.Equal(1.5f, f.Update(P(0.05f), true, 20).Score);           // 6th dip subtracts 1
    }

    [Fact]
    public void HoldAndRelease()
    {
        var f = Make(2.5f);
        f.Update(P(0.9f), true, 1); Assert.True(f.Update(P(0.9f), true, 2).TTL);
        now += 10; Assert.True(f.Update(P(0.0f), true, 3).TTL);           // still holding
        now += 50; var r = f.Update(P(0.0f), true, 4);
        Assert.False(r.TTL); Assert.Equal(RippleState.NoRipple, r.State);
    }

    [Fact]
    public void GateClosedResets()
    {
        var f = Make(100f);
        f.Update(P(0.9f), true, 1); f.Update(P(0.9f), true, 2);
        var r = f.Update(P(0.9f), false, 3);
        Assert.False(r.TTL); Assert.Equal(0f, r.Score); Assert.Equal(RippleState.NoRipple, r.State);
    }

    [Fact]
    public void DelayedTrigger()
    {
        var f = Make(2.5f); f.DelayMinMs = 100; f.DelayMaxMs = 100;
        f.Update(P(0.9f), true, 1);
        Assert.False(f.Update(P(0.9f), true, 2).TTL);                     // armed, not yet
        now += 50; Assert.False(f.Update(P(0.0f), true, 3).TTL);
        now += 60; var r = f.Update(P(0.0f), true, 4);
        Assert.True(r.TTL); Assert.Equal(100, r.TriggerDelayMs);
    }

    [Fact]
    public void EventCountSurvivesReset()
    {
        var f = Make(2.5f);
        f.Update(P(0.9f), true, 1); f.Update(P(0.9f), true, 2);
        Assert.Equal(1, f.EventCount);
        f.Reset();
        Assert.Equal(1, f.EventCount);
        Assert.Equal(RippleState.NoRipple, f.Update(P(0.0f), true, 3).State);
    }

    [Fact]
    public void ThresholdHoldTriggersAfterNConsecutive()
    {
        long t = 0;
        var s = new ThresholdHoldStage(0.5f, 3, 50, () => t);
        Assert.False(s.Update(P(0.6f), true, 1).TTL);
        Assert.False(s.Update(P(0.6f), true, 2).TTL);
        Assert.False(s.Update(P(0.4f), true, 3).TTL);                     // restarts the count
        Assert.False(s.Update(P(0.6f), true, 4).TTL);
        Assert.False(s.Update(P(0.6f), true, 5).TTL);
        var r = s.Update(P(0.6f), true, 6); Assert.True(r.TTL); Assert.Equal(1, r.EventCount);
        t += 10; Assert.True(s.Update(P(0.0f), true, 7).TTL);              // held
        t += 50; Assert.False(s.Update(P(0.0f), true, 8).TTL);
        Assert.Equal(1, s.EventCount);
    }
}
