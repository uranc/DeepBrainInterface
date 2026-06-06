using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("EMA Z-score with symmetric clipping. No history buffer — GC friendly.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class SlidingWindowZScore
    {
        [Description("EMA window size in samples (alpha = 2/(N+1)).")]
        public int WindowSize { get; set; } = 1250;

        [Description("Symmetric clip threshold (output clamped to ±Threshold).")]
        public float Threshold { get; set; } = 8f;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Defer(() =>
            {
                int channels = 0;
                double alpha = 0.0;
                double[] mu = null;
                double[] var = null;
                Mat outputMat = null;

                return source.Select(input =>
                {
                    int ch = input.Rows;
                    int nIn = input.Cols;

                    if (mu == null || channels != ch)
                    {
                        channels = ch;
                        alpha = 2.0 / (WindowSize + 1);
                        mu = new double[ch];
                        var = new double[ch];
                    }

                    if (outputMat == null || outputMat.Rows != ch || outputMat.Cols != nIn)
                        outputMat = new Mat(ch, nIn, Depth.F32, 1);

                    float threshold = Threshold;

                    unsafe
                    {
                        byte* inBytes = (byte*)input.Data.ToPointer();
                        byte* outBytes = (byte*)outputMat.Data.ToPointer();
                        int inStep = input.Step;
                        int outStep = outputMat.Step;

                        for (int t = 0; t < nIn; t++)
                        {
                            for (int c = 0; c < ch; c++)
                            {
                                float val = *(float*)(inBytes + c * inStep + t * sizeof(float));
                                double diff = val - mu[c];
                                mu[c] += alpha * diff;
                                var[c] = (1.0 - alpha) * (var[c] + alpha * diff * diff);
                                double sig = Math.Sqrt(var[c]);
                                double z = (val - mu[c]) / (sig + 1e-8);

                                // Symmetric clip
                                if (z > threshold) z = threshold;
                                else if (z < -threshold) z = -threshold;

                                *(float*)(outBytes + c * outStep + t * sizeof(float)) = (float)z;
                            }
                        }
                    }

                    return outputMat;
                });
            });
        }
    }
}