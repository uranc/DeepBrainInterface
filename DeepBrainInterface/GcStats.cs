using Bonsai;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    public struct GcStatsSample
    {
        public double TimeSeconds { get; set; }
        public int Gen0 { get; set; }
        public int Gen1 { get; set; }
        public int Gen2 { get; set; }
        public long HeapBytes { get; set; }
        public long PrivateBytes { get; set; }
        public long WorkingSetBytes { get; set; }
    }

    [Description("Cold telemetry: GC collection counts per generation and process memory, sampled on a timer thread. " +
                 "Log it next to the latency measurement to see whether stalls coincide with collections.")]
    public class GcStats : Source<GcStatsSample>
    {
        [Description("Sampling interval in milliseconds.")]
        public int IntervalMs { get; set; } = 100;

        public override IObservable<GcStatsSample> Generate()
        {
            return Observable.Defer(() =>
            {
                var proc = Process.GetCurrentProcess();
                var clock = Stopwatch.StartNew();
                var period = TimeSpan.FromMilliseconds(Math.Max(10, IntervalMs));
                return Observable.Interval(period)
                    .Select(_ =>
                    {
                        proc.Refresh();
                        return new GcStatsSample
                        {
                            TimeSeconds = clock.Elapsed.TotalSeconds,
                            Gen0 = GC.CollectionCount(0),
                            Gen1 = GC.CollectionCount(1),
                            Gen2 = GC.CollectionCount(2),
                            HeapBytes = GC.GetTotalMemory(false),
                            PrivateBytes = proc.PrivateMemorySize64,
                            WorkingSetBytes = proc.WorkingSet64
                        };
                    })
                    .Finally(proc.Dispose);
            });
        }
    }
}
