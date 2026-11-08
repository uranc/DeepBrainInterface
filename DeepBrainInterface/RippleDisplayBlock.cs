using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Display blocks for SendMatOverSocket without per-frame allocation: reorders all rows (DisplayOrder), keeps every " +
                 "DisplayDecimation-th sample, and emits F32 Mats of BlockSamples columns from a ring of RingDepth Mats allocated at start. " +
                 "Consumers must copy within RingDepth blocks (the socket sender serialises synchronously). 1.1.78.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDisplayBlock
    {
        [Description("Row order of the output (0-based input rows); empty = identity.")]
        public int[] DisplayOrder { get; set; } = new int[0];
        [Description("Keep every Nth input sample (1 = all).")]
        public int DisplayDecimation { get; set; } = 5;
        [Description("Samples per emitted block.")]
        public int BlockSamples { get; set; } = 30;
        [Description("Number of pre-allocated output Mats cycled through.")]
        public int RingDepth { get; set; } = 4;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Create<Mat>(observer =>
            {
                Mat[] ring = null; int ringIdx = 0, col = 0; long raw = 0; int[] order = null; int rows = 0;
                int dec = Math.Max(1, DisplayDecimation), block = Math.Max(1, BlockSamples), depth = Math.Max(2, RingDepth);
                return source.Subscribe(m =>
                {
                    try
                    {
                        if (ring == null || rows != m.Rows)
                        {
                            rows = m.Rows;
                            order = DisplayOrder != null && DisplayOrder.Length == rows ? (int[])DisplayOrder.Clone() : null;
                            if (order != null) foreach (var o in order) if (o < 0 || o >= rows) throw new InvalidOperationException("DisplayOrder index " + o + " outside " + rows + " rows");
                            ring = new Mat[depth]; for (int i = 0; i < depth; i++) ring[i] = new Mat(rows, block, Depth.F32, 1);
                            ringIdx = 0; col = 0;
                        }
                        unsafe
                        {
                            byte* src = (byte*)m.Data.ToPointer(); int sstep = m.Step; var depthIn = m.Depth;
                            for (int t = 0; t < m.Cols; t++)
                            {
                                if (raw++ % dec != 0) continue;
                                var dst = ring[ringIdx]; byte* d = (byte*)dst.Data.ToPointer(); int dstep = dst.Step;
                                for (int r = 0; r < rows; r++)
                                {
                                    int sr = order == null ? r : order[r]; byte* s = src + sr * sstep;
                                    float v = depthIn == Depth.F32 ? *(float*)(s + t * 4) : depthIn == Depth.U16 ? (float)*(ushort*)(s + t * 2) : depthIn == Depth.S16 ? (float)*(short*)(s + t * 2) : throw new InvalidOperationException("RippleDisplayBlock: depth must be U16, S16 or F32");
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
