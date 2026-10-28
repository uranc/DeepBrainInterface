using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Mirrors Buffer(1, DecimationFactor) + Buffer(WindowSize, Stride) applied to the LFP stream, " +
                 "but for the Rhd2164.Clock ulong[] hardware timestamp stream. " +
                 "Emits one ulong per LFP window: the hardware clock of the newest sample in that window. " +
                 "Zip this output with the detector output for exact per-window clock sync at full inference rate. " +
                 "Allocation-free (the earlier version copied a ulong[WindowSize] per window).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ClockSync
    {
        [Description("Must match DecimationFactor on the LFP stream (e.g. 12 for 30kHz -> 2.5kHz).")]
        public int DecimationFactor { get; set; } = 12;

        [Description("Must match the window size in Buffer(WindowSize, Stride) on the LFP stream.")]
        public int WindowSize { get; set; } = 44;

        [Description("Must match the stride in Buffer(WindowSize, Stride) on the LFP stream.")]
        public int Stride { get; set; } = 3;

        public IObservable<ulong> Process(IObservable<ulong[]> source)
        {
            return Observable.Create<ulong>(observer =>
            {
                int decimationFactor = Math.Max(1, DecimationFactor);
                int windowSize = Math.Max(1, WindowSize);
                int stride = Math.Max(1, Stride);

                // Counters start one short of their threshold so the very first raw sample is kept
                // (matching Buffer(1, Skip=DecimationFactor)) and the first full window emits at once
                // (matching Buffer(WindowSize, Stride)). Verified against Bonsai.Dsp.Buffer source.
                int decimCounter = decimationFactor - 1;
                int strideCounter = stride - 1;
                int filled = 0;

                return source.Subscribe(
                    block =>
                    {
                        try
                        {
                            for (int i = 0; i < block.Length; i++)
                            {
                                if (++decimCounter < decimationFactor) continue;
                                decimCounter = 0;

                                if (filled < windowSize) filled++;
                                if (filled == windowSize && ++strideCounter >= stride)
                                {
                                    strideCounter = 0;
                                    observer.OnNext(block[i]);
                                }
                            }
                        }
                        catch (Exception ex) { observer.OnError(ex); }
                    },
                    observer.OnError,
                    observer.OnCompleted);
            });
        }
    }
}
