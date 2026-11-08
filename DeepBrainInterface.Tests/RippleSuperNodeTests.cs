using System;
using System.Collections.Generic;
using System.IO;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using DeepBrainInterface;
using DeepBrainInterface.Core;
using OpenCV.Net;
using Xunit;

/// <summary>Runs on Windows (OpenCV.Net and ONNX Runtime natives). Build-checked on Linux.</summary>
public unsafe class RippleSuperNodeTests
{
    static string Model => Environment.GetEnvironmentVariable("DBI_TEST_MODEL") ?? @"D:\BonsaiModels\ES_model_tripletClean1450.onnx";
    static bool Have => File.Exists(Model);

    static RippleSuperNode Node() => new RippleSuperNode { Channels = new[] { 0, 1, 2, 3, 4, 5, 6, 7 }, DecimationFactor = 12, TimePoints = 44, Stride = 2, ModelPath = Model, ZScoreWindow = 1250 };

    static Tuple<Mat, ulong[]> Frame(int rows, int cols, int seed, ref ulong raw)
    {
        var m = new Mat(rows, cols, Depth.U16, 1); var rnd = new Random(seed); var clocks = new ulong[cols];
        ushort* p = (ushort*)m.Data.ToPointer(); int step = m.Step / 2;
        for (int t = 0; t < cols; t++) { clocks[t] = raw++; for (int r = 0; r < rows; r++) p[r * step + t] = (ushort)(32768 + 2000 * Math.Sin(0.05 * raw + r) + rnd.Next(-300, 300)); }
        return Tuple.Create(m, clocks);
    }

    [Fact]
    public void SecondSubscriptionThrows()
    {
        if (!Have) return;
        var node = Node(); var src = new Subject<Tuple<Mat, ulong[]>>();
        var obs = node.Process(src);
        using (obs.Subscribe(_ => { }))
        {
            Exception err = null;
            using (obs.Subscribe(_ => { }, e => err = e)) { }
            Assert.NotNull(err); Assert.Contains("subscription", err.Message, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ChannelOutOfRangeErrors()
    {
        if (!Have) return;
        var node = Node(); node.Channels = new[] { 9 };
        var src = new Subject<Tuple<Mat, ulong[]>>(); Exception err = null;
        using (node.Process(src).Subscribe(_ => { }, e => err = e))
        {
            ulong raw = 0; src.OnNext(Frame(8, 24, 1, ref raw));
        }
        Assert.NotNull(err); Assert.Contains("Channels", err.Message);
    }

    [Fact]
    public void OutputMatchesFrameProcessor()
    {
        if (!Have) return;
        var node = Node(); var src = new Subject<Tuple<Mat, ulong[]>>(); var viaNode = new List<RippleOut>();
        var cfg = new ProcessorConfig { Channels = new[] { 0, 1, 2, 3, 4, 5, 6, 7 }, DecimationFactor = 12, TimePoints = 44, Stride = 2, ModelPaths = new[] { Model }, StageFactory = p => new IPreprocessStage[] { new ZScoreStage(8, 1250) } };
        var direct = new List<RippleOut>(); var buf = new RippleOut[24];
        using (var fp = new FrameProcessor(cfg, new RippleFsm()))
        using (node.Process(src).Subscribe(viaNode.Add))
        {
            ulong raw = 0;
            for (int f = 0; f < 60; f++)
            {
                var fr = Frame(8, 24, f, ref raw);
                src.OnNext(fr);
                fixed (ulong* ck = fr.Item2) { int n = fp.ProcessFrame((ushort*)fr.Item1.Data.ToPointer(), 8, 24, fr.Item1.Step, ck, true, buf); for (int i = 0; i < n; i++) direct.Add(buf[i]); }
            }
        }
        Assert.Equal(direct.Count, viaNode.Count);
        for (int i = 0; i < direct.Count; i++) { Assert.Equal(direct[i].Clock, viaNode[i].Clock); Assert.Equal(direct[i].Prob0, viaNode[i].Prob0); Assert.Equal(direct[i].TTL, viaNode[i].TTL); }
    }

    [Fact]
    public void DisplayBlocksComeFromRing()
    {
        var disp = new RippleDisplayBlock { DisplayDecimation = 1, BlockSamples = 24, RingDepth = 4 };
        var src = new Subject<Mat>(); var blocks = new List<Mat>();
        using (disp.Process(src).Subscribe(blocks.Add))
        {
            ulong raw = 0;
            for (int f = 0; f < 5; f++) src.OnNext(Frame(64, 24, f, ref raw).Item1);
        }
        Assert.Equal(5, blocks.Count);
        Assert.NotSame(blocks[0], blocks[1]); Assert.NotSame(blocks[0], blocks[2]); Assert.NotSame(blocks[0], blocks[3]);
        Assert.Same(blocks[0], blocks[4]);
        Assert.Equal(64, blocks[0].Rows); Assert.Equal(24, blocks[0].Cols); Assert.Equal(Depth.F32, blocks[0].Depth);
    }
}
