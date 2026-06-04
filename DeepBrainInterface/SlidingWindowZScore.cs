using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Exact rectangular sliding-window Z-score. Matches offline Python implementation.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class SlidingWindowZScore
    {
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        private const uint _MCW_DN = 0x03000000;
        private const uint _DN_FLUSH = 0x01000000;

        [Description("Window length in samples (matches offline win=1250).")]
        public int WindowSize { get; set; } = 1250;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Defer(() =>
            {
                int      channels  = 0;
                float[]  history   = null;   // ring buffer [head * channels + c]
                double[] sum       = null;   // per-channel running sum,  float64
                double[] sumSq     = null;   // per-channel running sum², float64
                int      head      = 0;
                int      count     = 0;
                bool     flushed   = false;
                Mat      outputMat = null;

                return source.Select(input =>
                {
                    if (!flushed)
                    {
                        try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { }
                        flushed = true;
                    }

                    int ch  = input.Rows;
                    int nIn = input.Cols;

                    if (history == null || channels != ch)
                    {
                        channels = ch;
                        history  = new float[WindowSize * channels];
                        sum      = new double[channels];
                        sumSq    = new double[channels];
                        head     = 0;
                        count    = 0;
                    }

                    if (outputMat == null || outputMat.Rows != ch || outputMat.Cols != nIn)
                        outputMat = new Mat(ch, nIn, Depth.F32, 1);

                    unsafe
                    {
                        // Use step-based pointer arithmetic — correct for both contiguous
                        // and non-contiguous Mats (e.g. sub-matrix slices from dsp:Buffer).
                        byte* inBytes  = (byte*)input.Data.ToPointer();
                        byte* outBytes = (byte*)outputMat.Data.ToPointer();
                        int   inStep   = input.Step;
                        int   outStep  = outputMat.Step;

                        for (int t = 0; t < nIn; t++)
                        {
                            bool full = count >= WindowSize;

                            for (int c = 0; c < channels; c++)
                            {
                                float  val     = *(float*)(inBytes  + c * inStep  + t * sizeof(float));
                                double dropped = full ? (double)history[head * channels + c] : 0.0;

                                sum[c]   += val - dropped;
                                sumSq[c] += (double)val * val - dropped * dropped;
                                history[head * channels + c] = val;

                                // n includes current sample — matches Python L = min(t+1, win)
                                int    n   = full ? WindowSize : count + 1;
                                double mu  = sum[c] / n;
                                double var = Math.Max(sumSq[c] / n - mu * mu, 0.0);
                                double sig = Math.Sqrt(var);

                                // matches Python: (x - mu) / (sig + eps), eps = 1e-8
                                *(float*)(outBytes + c * outStep + t * sizeof(float)) =
                                    (float)((val - mu) / (sig + 1e-8));
                            }

                            head = (head + 1) % WindowSize;
                            if (count < WindowSize) count++;
                        }
                    }

                    // Safe to return without Clone(): downstream dsp:ConvertScale
                    // always creates its own output Mat before any async consumer sees this.
                    return outputMat;
                });
            });
        }
    }
}
