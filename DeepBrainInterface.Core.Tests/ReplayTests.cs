using System;
using System.Collections.Generic;
using System.IO;
using DeepBrainInterface.Core;
using Xunit;
using Xunit.Abstractions;

/// <summary>Replays the first 120 s of session 04112026 suffix 2 through FrameProcessor with the deployed configuration and
/// compares with the probabilities Bonsai wrote online (ripple_out). Files from SWRProject/matlab/export_replay.py.</summary>
public unsafe class ReplayTests
{
    readonly ITestOutputHelper log;
    public ReplayTests(ITestOutputHelper log) { this.log = log; }
    static string Dir => Environment.GetEnvironmentVariable("DBI_REPLAY_DIR") ?? "/home/uranc/Documents/SWRProject/report/replay";
    static string Raw => Path.Combine(Dir, "replay_04112026_2_raw.bin");
    static string Clk => Path.Combine(Dir, "replay_04112026_2_clock.bin");
    static string Dec => Path.Combine(Dir, "replay_04112026_2_decisions.csv");
    static string Model => Path.Combine(Dir, "model_tripletClean1450.onnx");
    const int CH = 8, FRAME = 24; const double ACQ_HZ = 250e6;

    static ProcessorConfig DeployedConfig() => new ProcessorConfig
    {
        Channels = new[] { 0, 1, 2, 3, 4, 5, 6, 7 }, DecimationFactor = 12, PhaseCount = 1, TimePoints = 44, Stride = 2,
        ModelPaths = new[] { Model }, StageFactory = p => new IPreprocessStage[] { new ZScoreStage(CH, 1250) },
    };
    static RippleFsm DeployedFsm() => new RippleFsm { Threshold1 = 0.6f, Threshold2 = 0.7f, Threshold3 = 0.8f, TargetEvidenceScore = 10f, FramesToWaitBeforeDecay = 5, ScoreDecayPerFrame = 1f, PostRippleMs = 250, DetectionEnabled = true, HardwareClockHz = ACQ_HZ };

    bool Have()
    {
        foreach (var f in new[] { Raw, Clk, Dec, Model }) if (!File.Exists(f)) { log.WriteLine("skipped: missing " + f); return false; }
        return true;
    }

    /// <summary>Feeds the recording in 24-sample frames (the recorded BufferSize) and returns the decisions.</summary>
    static List<RippleOut> Replay(FrameProcessor fp, short[] raw, ulong[] clock, int maxSamples)
    {
        var outs = new List<RippleOut>(); var buf = new RippleOut[FRAME];
        var frame = new ushort[CH * FRAME]; var ck = new ulong[FRAME];
        int n = Math.Min(maxSamples, clock.Length) / FRAME * FRAME;
        fixed (ushort* d = frame) fixed (ulong* c = ck)
        {
            for (int s = 0; s < n; s += FRAME)
            {
                for (int t = 0; t < FRAME; t++) { ck[t] = clock[s + t]; for (int ch = 0; ch < CH; ch++) frame[ch * FRAME + t] = (ushort)raw[(s + t) * CH + ch]; }
                int k = fp.ProcessFrame(d, CH, FRAME, FRAME * sizeof(ushort), c, true, buf);
                for (int i = 0; i < k; i++) outs.Add(buf[i]);
            }
        }
        return outs;
    }

    static (short[] raw, ulong[] clock) Load()
    {
        var rb = File.ReadAllBytes(Raw); var raw = new short[rb.Length / 2]; Buffer.BlockCopy(rb, 0, raw, 0, rb.Length);
        var cb = File.ReadAllBytes(Clk); var clock = new ulong[cb.Length / 8]; Buffer.BlockCopy(cb, 0, clock, 0, cb.Length);
        return (raw, clock);
    }

    [Fact]
    public void ReplayMatchesRecordedProbabilities()
    {
        if (!Have()) return;
        var (raw, clock) = Load();
        var rec = new Dictionary<ulong, (float prob, bool ttl)>();
        foreach (var line in File.ReadLines(Dec)) { if (line.StartsWith("clock")) continue; var f = line.Split(','); rec[ulong.Parse(f[0])] = (float.Parse(f[1], System.Globalization.CultureInfo.InvariantCulture), f[2] == "1"); }
        List<RippleOut> outs;
        using (var fp = new FrameProcessor(DeployedConfig(), DeployedFsm())) outs = Replay(fp, raw, clock, raw.Length / CH);
        Assert.True(outs.Count > 100000, "too few decisions: " + outs.Count);
        Assert.Equal(9850568UL, outs[0].Clock);                       // the recorded first decision clock
        int joined = 0, exact = 0, close = 0, warm = 0; float maxd = 0; var mism = new List<string>();
        int skip = 1250 / 2;                                           // z-score window not yet full: 1250 decimated samples = 625 decisions
        for (int i = 0; i < outs.Count; i++)
        {
            if (!rec.TryGetValue(outs[i].Clock, out var r)) continue;
            joined++; if (i < skip) { warm++; continue; }
            float d = Math.Abs(outs[i].Prob0 - r.prob); if (d > maxd) maxd = d;
            if (d == 0f) exact++; else if (d < 1e-6f) close++; else if (mism.Count < 5) mism.Add($"i={i} clock={outs[i].Clock} replay={outs[i].Prob0:R} recorded={r.prob:R}");
        }
        log.WriteLine($"decisions {outs.Count}, joined {joined} (warm-up {warm}), exact {exact}, <1e-6 {close}, max |d| {maxd:E2}");
        foreach (var m in mism) log.WriteLine("  " + m);
        Assert.True(joined >= outs.Count - 2, $"only {joined} of {outs.Count} decisions found in the recording");
        Assert.True((exact + close) >= 0.999 * (joined - warm), $"exact {exact} + close {close} of {joined - warm}: max |d| = {maxd}");
        // every recorded TTL rise must be a TTL rise in the replay
        var replayRises = new HashSet<ulong>(); bool prev = false;
        foreach (var o in outs) { if (o.TTL && !prev) replayRises.Add(o.Clock); prev = o.TTL; }
        int recRises = 0, matched = 0; bool prevRec = false;
        foreach (var o in outs) { if (!rec.TryGetValue(o.Clock, out var r)) continue; if (r.ttl && !prevRec) { recRises++; if (replayRises.Contains(o.Clock)) matched++; } prevRec = r.ttl; }
        log.WriteLine($"recorded TTL rises {recRises}, matched in replay {matched}, replay rises {replayRises.Count}");
        // extra replay rises: distance to the nearest recorded rise (hold boundaries: wall clock online vs hardware clock here)
        var recRiseList = new List<ulong>(); prevRec = false;
        foreach (var o in outs) { if (rec.TryGetValue(o.Clock, out var r)) { if (r.ttl && !prevRec) recRiseList.Add(o.Clock); prevRec = r.ttl; } }
        foreach (var c in replayRises)
        {
            if (recRiseList.Contains(c)) continue;
            double best = double.MaxValue; foreach (var rc in recRiseList) best = Math.Min(best, Math.Abs((double)c - rc));
            log.WriteLine($"  extra replay rise at clock {c}: nearest recorded rise {best / ACQ_HZ * 1e3:F1} ms away");
        }
        Assert.True(recRises > 0); Assert.Equal(recRises, matched);
    }

    [Fact]
    public void AllocationFreeAfterWarmup()
    {
        if (!Have()) return;
        var (raw, clock) = Load();
        using (var fp = new FrameProcessor(DeployedConfig(), DeployedFsm()))
        {
            int warm = 29346 * 5 / FRAME * FRAME;                        // 5 s
            Replay(fp, raw, clock, warm);
            var buf = new RippleOut[FRAME]; var frame = new ushort[CH * FRAME]; var ck = new ulong[FRAME];
            long before = GC.GetAllocatedBytesForCurrentThread();
            fixed (ushort* d = frame) fixed (ulong* c = ck)
            {
                for (int s = warm; s < warm + 29346 * 10 / FRAME * FRAME; s += FRAME)
                {
                    for (int t = 0; t < FRAME; t++) { ck[t] = clock[s + t]; for (int ch = 0; ch < CH; ch++) frame[ch * FRAME + t] = (ushort)raw[(s + t) * CH + ch]; }
                    fp.ProcessFrame(d, CH, FRAME, FRAME * sizeof(ushort), c, true, buf);
                }
            }
            long after = GC.GetAllocatedBytesForCurrentThread();
            log.WriteLine($"allocated during 10 s of frames: {after - before} bytes");
            Assert.True(after - before < 64 * 1024, $"allocated {after - before} bytes in 10 s");
        }
    }
}
