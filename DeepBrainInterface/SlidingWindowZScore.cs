using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Applies a running Z-Score using a pure mathematical counter. Strict F32 input. Zero ring buffers.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class SlidingWindowZScore
    {
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        private const uint _MCW_DN = 0x03000000;
        private const uint _DN_FLUSH = 0x01000000;

        [Description("The history counter limit used to weight the running mean and variance. Defaults to 1250.")]
        public int WindowSize { get; set; } = 1250;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Defer(() =>
            {
                float[] means = null;
                float[] variances = null;
                int count = 0;
                Mat outputMat = null;
                bool denormalFlushed = false;

                return source.Select(input =>
                {
                    // Flush subnormals on first call — must run on the acquisition thread, not subscription thread.
                    if (!denormalFlushed)
                    {
                        try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { }
                        denormalFlushed = true;
                    }

                    int channels = input.Rows;
                    int incomingSamples = input.Cols;

                    if (means == null)
                    {
                        means = new float[channels];
                        variances = new float[channels];
                        count = 0;
                    }

                    if (outputMat == null || outputMat.Cols != incomingSamples)
                        outputMat = new Mat(channels, incomingSamples, Depth.F32, 1);

                    unsafe
                    {
                        float* inData = (float*)input.Data.ToPointer();
                        float* outData = (float*)outputMat.Data.ToPointer();

                        for (int t = 0; t < incomingSamples; t++)
                        {
                            if (count < WindowSize) count++;
                            float alpha = 1.0f / count;
                            float oneMinusAlpha = 1.0f - alpha;

                            for (int c = 0; c < channels; c++)
                            {
                                int idx = c * incomingSamples + t;
                                float val = inData[idx];

                                float diff = val - means[c];
                                means[c] += alpha * diff;
                                variances[c] = oneMinusAlpha * (variances[c] + alpha * diff * diff);

                                float std = (float)Math.Sqrt(variances[c]);
                                if (std < 1e-6f) std = 1f;

                                outData[idx] = (val - means[c]) / std;
                            }
                        }
                    }

                    // No Clone() — downstream dsp:ConvertScale creates its own output Mat,
                    // so outputMat is not retained by any async consumer and is safe to reuse.
                    return outputMat;
                });
            });
        }
    }
}