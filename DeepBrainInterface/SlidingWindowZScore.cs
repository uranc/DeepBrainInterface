using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Exact rectangular sliding-window Z-score matching offline training (float64 accumulators). " +
                 "Set DecimationFactor=12 and remove the upstream dsp:Buffer(Count=1,Skip=12) node.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class SlidingWindowZScore
    {
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        private const uint _MCW_DN = 0x03000000;
        private const uint _DN_FLUSH = 0x01000000;

        [Description("Rectangular window length in decimated samples (matches offline win=1250).")]
        public int WindowSize { get; set; } = 1250;

        [Description("Downsample factor applied before Z-scoring. " +
                     "Set to 12 and remove dsp:Buffer(Count=1,Skip=12) upstream to combine both operations.")]
        public int DecimationFactor { get; set; } = 1;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Create<Mat>(observer =>
            {
                int    channels     = 0;
                float[] history     = null;   // circular ring [head * channels + c], float32
                double[] sum        = null;   // per-channel running sum,  float64
                double[] sumSq      = null;   // per-channel running sum², float64
                int    head         = 0;      // next write position in ring
                int    count        = 0;      // decimated samples seen, capped at WindowSize
                int    decimCounter = 0;
                bool   flushed      = false;
                Mat    outputMat    = null;   // single pre-allocated 1-column output, reused every emission

                var sub = source.Subscribe(
                    input =>
                    {
                        // Must run on the acquisition thread — deferred from subscription time.
                        if (!flushed)
                        {
                            try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { }
                            flushed = true;
                        }

                        int ch  = input.Rows;
                        int nIn = input.Cols;
                        int df  = DecimationFactor < 1 ? 1 : DecimationFactor;

                        if (history == null || channels != ch)
                        {
                            channels    = ch;
                            history     = new float[WindowSize * channels];
                            sum         = new double[channels];
                            sumSq       = new double[channels];
                            outputMat   = new Mat(channels, 1, Depth.F32, 1);
                            head = 0; count = 0; decimCounter = 0;
                        }

                        unsafe
                        {
                            float* inData  = (float*)input.Data.ToPointer();
                            float* outData = (float*)outputMat.Data.ToPointer();

                            for (int t = 0; t < nIn; t++)
                            {
                                if (++decimCounter < df) continue;
                                decimCounter = 0;

                                bool full = count >= WindowSize;

                                // Anti-drift: exact recompute every WindowSize decimated samples
                                // (once per ring revolution, ~every 500ms at 2500Hz). Prevents
                                // float64 accumulation error from drifting over long recordings.
                                if (full && head == 0)
                                {
                                    for (int c = 0; c < channels; c++) { sum[c] = 0.0; sumSq[c] = 0.0; }
                                    for (int i = 0; i < WindowSize; i++)
                                        for (int c = 0; c < channels; c++)
                                        {
                                            double v = history[i * channels + c];
                                            sum[c] += v; sumSq[c] += v * v;
                                        }
                                }

                                for (int c = 0; c < channels; c++)
                                {
                                    float  val     = inData[c * nIn + t];
                                    double dropped = full ? (double)history[head * channels + c] : 0.0;

                                    sum[c]   += val - dropped;
                                    sumSq[c] += (double)val * val - dropped * dropped;
                                    history[head * channels + c] = val;

                                    // n includes current sample, matching Python's L = min(t+1, win).
                                    int    n   = full ? WindowSize : count + 1;
                                    double mu  = sum[c] / n;
                                    double var = Math.Max(sumSq[c] / n - mu * mu, 0.0);
                                    double sig = Math.Sqrt(var);
                                    outData[c] = (float)((val - mu) / (sig + 1e-8));
                                }

                                head = (head + 1) % WindowSize;
                                if (count < WindowSize) count++;

                                // outputMat is safe to reuse: downstream dsp:ConvertScale
                                // creates its own copy before any async consumer sees the data.
                                observer.OnNext(outputMat);
                            }
                        }
                    },
                    observer.OnError,
                    observer.OnCompleted
                );

                return new CompositeDisposable(sub, outputMat == null
                    ? Disposable.Empty
                    : Disposable.Create(() => outputMat.Dispose()));
            });
        }
    }
}
