using System;
using System.Collections.Generic;
using System.IO;
using DeepBrainInterface.Core;
using Xunit;

public unsafe class FrameProcessorTests
{
    static string Model => Environment.GetEnvironmentVariable("DBI_TEST_MODEL") ?? "/home/uranc/Documents/SWRProject/pi_streaming/model_tripletRemake1334_1Batch.onnx";
    static bool Have => File.Exists(Model);

    static ProcessorConfig Cfg(int phases = 1, params string[] models) => new ProcessorConfig
    {
        Channels = new[] { 0, 1, 2, 3, 4, 5, 6, 7 }, DecimationFactor = 12, PhaseCount = phases, TimePoints = 44, Stride = 2,
        ModelPaths = models.Length > 0 ? models : new[] { Model }, StageFactory = p => new IPreprocessStage[] { new ZScoreStage(8, 1250) },
    };

    /// <summary>Feeds nFrames frames of 8 rows x 24 columns (clock = raw sample index) and returns all decisions.</summary>
    static List<RippleOut> Run(FrameProcessor fp, int nFrames, int seed = 1)
    {
        var rnd = new Random(seed); var outs = new List<RippleOut>(); var buf = new RippleOut[24];
        var data = new ushort[8 * 24]; var clocks = new ulong[24]; ulong raw = 0;
        for (int f = 0; f < nFrames; f++)
        {
            for (int t = 0; t < 24; t++) { clocks[t] = raw++; for (int c = 0; c < 8; c++) data[c * 24 + t] = (ushort)(32768 + 2000 * Math.Sin(0.05 * raw + c) + rnd.Next(-300, 300)); }
            fixed (ushort* d = data) fixed (ulong* ck = clocks)
            {
                int n = fp.ProcessFrame(d, 8, 24, 24 * sizeof(ushort), ck, true, buf);
                for (int i = 0; i < n; i++) outs.Add(buf[i]);
            }
        }
        return outs;
    }

    [Fact]
    public void FirstDecisionAfterTimePointsSamplesThenEveryStride()
    {
        if (!Have) return;
        using (var fp = new FrameProcessor(Cfg(), new RippleFsm()))
        {
            var outs = Run(fp, 40);
            Assert.Equal(43UL * 12, outs[0].Clock);           // 44th kept sample: raw index 43*12 (decimation keeps the first sample)
            Assert.Equal(45UL * 12, outs[1].Clock);           // stride 2 decimated samples = 24 raw
            Assert.Equal(47UL * 12, outs[2].Clock);
            Assert.Equal(0L, outs[0].DecisionIndex); Assert.Equal(2L, outs[2].DecisionIndex);
            Assert.InRange(outs[0].Prob0, 0f, 1f); Assert.True(float.IsNaN(outs[0].Prob1));
        }
    }

    [Fact]
    public void PhaseCountOverDecimationRejected()
    {
        if (!Have) return;
        var ex = Assert.Throws<ArgumentException>(() => new FrameProcessor(Cfg(13), new RippleFsm()));
        Assert.Contains("PhaseCount", ex.Message);
    }

    [Fact]
    public void TwelvePhasesPhaseZeroEqualsSinglePhase()
    {
        if (!Have) return;
        List<RippleOut> a, b;
        using (var fp = new FrameProcessor(Cfg(1), new RippleFsm())) a = Run(fp, 60);
        using (var fp = new FrameProcessor(Cfg(12), new RippleFsm())) b = Run(fp, 60);
        Assert.Equal(a.Count, b.Count);
        for (int i = 0; i < a.Count; i++) { Assert.Equal(a[i].Clock, b[i].Clock); Assert.Equal(a[i].Prob0, b[i].Prob0); }
    }

    [Fact]
    public void BothModelsSameFileGiveEqualProbs()
    {
        if (!Have) return;
        var cfg = Cfg(1, Model, Model); cfg.ActiveModels = 2;
        using (var fp = new FrameProcessor(cfg, new RippleFsm()))
        {
            var outs = Run(fp, 40);
            Assert.NotEmpty(outs);
            foreach (var o in outs) { Assert.False(float.IsNaN(o.Prob1)); Assert.Equal(o.Prob0, o.Prob1); }
        }
    }

    [Fact]
    public void LiveChangeBumpsConfigVersionOnce()
    {
        if (!Have) return;
        var cfg = Cfg(1, Model, Model);
        using (var fp = new FrameProcessor(cfg, new RippleFsm()))
        {
            var a = Run(fp, 30); int v0 = a[a.Count - 1].ConfigVersion;
            fp.SetActiveModels(1); fp.SetCombiner(Combiner.Max);
            var b = Run(fp, 10);
            Assert.Equal(v0 + 2, b[0].ConfigVersion); Assert.Equal(v0 + 2, b[b.Count - 1].ConfigVersion);
        }
    }
}
