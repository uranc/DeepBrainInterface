using System;
using System.IO;
using DeepBrainInterface.Core;
using Xunit;

public class OrtRunnerTests
{
    static string Model => Environment.GetEnvironmentVariable("DBI_TEST_MODEL") ?? "/home/uranc/Documents/SWRProject/pi_streaming/model_tripletRemake1334_1Batch.onnx";
    static string MultiOut => Path.Combine(Path.GetDirectoryName(Model), "stream_B3_fused.onnx");
    static bool Have => File.Exists(Model);

    [Fact]
    public void ReadsShapeFromModel()
    {
        if (!Have) { Console.WriteLine("skipped: model missing " + Model); return; }
        var (b, t, c) = OrtRunner.ReadInputShape(Model);
        Assert.Equal(44, t); Assert.Equal(8, c); Assert.True(b == -1 || b == 1);
    }

    [Fact]
    public void RejectsWrongChannels()
    {
        if (!Have) return;
        var ex = Assert.Throws<ArgumentException>(() => new OrtRunner(Model, 1, 44, 7));
        Assert.Contains("channels", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RejectsMultiOutputModel()
    {
        if (!File.Exists(MultiOut)) return;
        var ex = Assert.Throws<ArgumentException>(() => new OrtRunner(MultiOut, 1, 3, 8));
        Assert.Contains("output", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ZeroInputGivesStableProbability()
    {
        if (!Have) return;
        using (var r = new OrtRunner(Model, 1, 44, 8))
        {
            Array.Clear(r.Input, 0, r.Input.Length);
            r.Run(); float p1 = r.Output[0]; r.Run(); float p2 = r.Output[0];
            Assert.Equal(p1, p2); Assert.InRange(p1, 0f, 1f);
        }
    }

    [Fact]
    public void BatchRowsIndependent()
    {
        if (!Have) return;
        float pZeroSingle;
        using (var r1 = new OrtRunner(Model, 1, 44, 8)) { Array.Clear(r1.Input, 0, r1.Input.Length); r1.Run(); pZeroSingle = r1.Output[0]; }
        if (OrtRunner.ReadInputShape(Model).batch == 1)
        {   // fixed-batch export: a 2-row runner must be refused with a message naming the batch
            var ex = Assert.Throws<ArgumentException>(() => new OrtRunner(Model, 2, 44, 8));
            Assert.Contains("batch", ex.Message, StringComparison.OrdinalIgnoreCase);
            return;
        }
        using (var r2 = new OrtRunner(Model, 2, 44, 8))
        {
            Array.Clear(r2.Input, 0, r2.Input.Length);
            for (int t = 0; t < 44; t++) for (int c = 0; c < 8; c++) r2.Input[t * 8 + c] = (float)Math.Sin(0.3 * t + c);   // row 0 = sine, row 1 = zeros
            r2.Run();
            Assert.Equal(pZeroSingle, r2.Output[1]);
            Assert.NotEqual(r2.Output[0], r2.Output[1]);
        }
    }
}
