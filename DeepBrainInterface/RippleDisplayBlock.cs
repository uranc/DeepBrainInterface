using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Display blocks for SendMatOverSocket without per-frame allocation: reorders the rows by ChannelMap (probe order), picks the rows " +
                 "listed in Channels (indices into the mapped order; empty = all), keeps every DisplayDecimation-th sample, and emits F32 Mats of " +
                 "BlockSamples columns from a ring of RingDepth Mats allocated at start. Consumers must copy within RingDepth blocks (the socket " +
                 "sender serialises synchronously). 1.1.80.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDisplayBlock
    {
        [TypeConverter(typeof(UnidimensionalArrayConverter))]
        [Description("Comma-separated 0-based input rows in display order (the probe map, e.g. 42,63,48,...). Must list every row once; empty = input order.")]
        public int[] ChannelMap { get; set; } = new int[0];
        [TypeConverter(typeof(UnidimensionalArrayConverter))]
        [Description("Comma-separated indices INTO THE MAPPED ORDER to show, e.g. 41,40,39,38,37,36,35,34 = eight contacts of the sorted probe. " +
                     "Empty = all rows. Output rows = this many, in this order.")]
        public int[] Channels { get; set; } = new int[0];
        [Description("Keep every Nth input sample (1 = all; 5 = 500 Hz display from 2.5 kHz, 60 = 500 Hz from 30 kHz).")]
        public int DisplayDecimation { get; set; } = 5;
        [Description("Samples per emitted block.")]
        public int BlockSamples { get; set; } = 30;
        [Description("Number of pre-allocated output Mats cycled through.")]
        public int RingDepth { get; set; } = 4;

        /// <summary>Resolves ChannelMap and Channels against an input with `rows` rows: returns the input row for each output row.</summary>
        public static int[] ResolveRows(int[] channelMap, int[] channels, int rows)
        {
            int[] mapped;
            if (channelMap == null || channelMap.Length == 0) { mapped = new int[rows]; for (int i = 0; i < rows; i++) mapped[i] = i; }
            else
            {
                if (channelMap.Length != rows) throw new InvalidOperationException("ChannelMap lists " + channelMap.Length + " rows, the input has " + rows);
                var seen = new bool[rows];
                foreach (var o in channelMap)
                {
                    if (o < 0 || o >= rows) throw new InvalidOperationException("ChannelMap index " + o + " outside " + rows + " rows");
                    if (seen[o]) throw new InvalidOperationException("ChannelMap lists row " + o + " twice");
                    seen[o] = true;
                }
                mapped = (int[])channelMap.Clone();
            }
            if (channels == null || channels.Length == 0) return mapped;
            var outRows = new int[channels.Length];
            for (int i = 0; i < channels.Length; i++)
            {
                int k = channels[i];
                if (k < 0 || k >= mapped.Length) throw new InvalidOperationException("Channels index " + k + " outside the " + mapped.Length + " mapped rows");
                outRows[i] = mapped[k];
            }
            return outRows;
        }

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Create<Mat>(observer =>
            {
                Mat[] ring = null; int ringIdx = 0, col = 0; long raw = 0; int[] srcRow = null; int inRows = 0, outRows = 0;
                int dec = Math.Max(1, DisplayDecimation), block = Math.Max(1, BlockSamples), depth = Math.Max(2, RingDepth);
                return source.Subscribe(m =>
                {
                    try
                    {
                        if (ring == null || inRows != m.Rows)
                        {
                            inRows = m.Rows;
                            srcRow = ResolveRows(ChannelMap, Channels, inRows); outRows = srcRow.Length;
                            // old ring left to the GC: a consumer may still hold one of its blocks
                            ring = new Mat[depth]; for (int i = 0; i < depth; i++) ring[i] = new Mat(outRows, block, Depth.F32, 1);
                            ringIdx = 0; col = 0;
                        }
                        unsafe
                        {
                            byte* src = (byte*)m.Data.ToPointer(); int sstep = m.Step; var depthIn = m.Depth;
                            if (depthIn != Depth.F32 && depthIn != Depth.U16 && depthIn != Depth.S16) throw new InvalidOperationException("RippleDisplayBlock: depth must be U16, S16 or F32, got " + depthIn);
                            for (int t = 0; t < m.Cols; t++)
                            {
                                if (raw++ % dec != 0) continue;
                                var dst = ring[ringIdx]; byte* d = (byte*)dst.Data.ToPointer(); int dstep = dst.Step;
                                for (int r = 0; r < outRows; r++)
                                {
                                    byte* s = src + srcRow[r] * sstep;
                                    float v = depthIn == Depth.F32 ? *(float*)(s + t * 4) : depthIn == Depth.U16 ? (float)*(ushort*)(s + t * 2) : (float)*(short*)(s + t * 2);
                                    *(float*)(d + r * dstep + col * 4) = v;
                                }
                                if (++col == block) { col = 0; observer.OnNext(dst); ringIdx = ringIdx + 1 == depth ? 0 : ringIdx + 1; }
                            }
                        }
                    }
                    catch (Exception ex) { observer.OnError(ex); }
                }, observer.OnError, observer.OnCompleted);
            });
        }
    }
}
