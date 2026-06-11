using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Mirrors Buffer(WindowSize, Stride) + Buffer(1, DecimationFactor) applied to the LFP stream, " +
                 "but for the Rhd2164.Clock ulong[] hardware timestamp stream. " +
                 "Emits ulong[WindowSize] in sync with each LFP Buffer window. " +
                 "Use clock[WindowSize-1] downstream for the newest sample timestamp in each window. " +
                 "Zip this output with the LFP Buffer output for exact per-window clock sync at full inference rate.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ClockSync
    {
        [Description("Must match DecimationFactor on the LFP stream (e.g. 12 for 30kHz -> 2.5kHz).")]
        public int DecimationFactor { get; set; } = 12;

        [Description("Must match the window size in Buffer(WindowSize, Stride) on the LFP stream.")]
        public int WindowSize { get; set; } = 44;

        [Description("Must match the stride in Buffer(WindowSize, Stride) on the LFP stream.")]
        public int Stride { get; set; } = 3;

        public IObservable<ulong[]> Process(IObservable<ulong[]> source)
        {
            return Observable.Defer(() =>
            {
                int decimCounter  = 0;
                int strideCounter = 0;
                ulong[] window    = new ulong[WindowSize];
                int     filled    = 0;       // how many decimated samples are in the window so far

                return source.SelectMany(block =>
                {
                    var emissions = new System.Collections.Generic.List<ulong[]>();

                    foreach (ulong tick in block)
                    {
                        // 1. Decimate: keep every DecimationFactor-th raw sample.
                        decimCounter++;
                        if (decimCounter < DecimationFactor) continue;
                        decimCounter = 0;

                        // 2. Slide window left, append newest decimated sample at the end.
                        if (filled < WindowSize)
                        {
                            window[filled] = tick;
                            filled++;
                        }
                        else
                        {
                            Buffer.BlockCopy(window, sizeof(ulong), window, 0, (WindowSize - 1) * sizeof(ulong));
                            window[WindowSize - 1] = tick;
                        }

                        // 3. Emit a copy of the window every Stride decimated samples.
                        if (filled == WindowSize)
                        {
                            strideCounter++;
                            if (strideCounter >= Stride)
                            {
                                strideCounter = 0;
                                ulong[] copy = new ulong[WindowSize];
                                Buffer.BlockCopy(window, 0, copy, 0, WindowSize * sizeof(ulong));
                                emissions.Add(copy);
                            }
                        }
                    }

                    return emissions.ToObservable();
                });
            });
        }
    }
}
