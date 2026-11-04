using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Exact Rectangular Sliding Window Z-score. Matches Python training environment.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class SlidingWindowZScore
    {
        [Description("Exact window size in samples.")]
        public int WindowSize { get; set; } = 1250;

        [Description("Epsilon to prevent division by zero.")]
        public float Eps { get; set; } = 1e-8f;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Defer(() =>
            {
                int channels = 0;
                double[] sumX = null;
                double[] sumX2 = null;
                float[,] history = null;
                int historyIndex = 0;
                long sampleCount = 0;
                Mat outputMat = null;

                return source.Select(input =>
                {
                    int ch = input.Rows;
                    int nIn = input.Cols;

                    // Initialize state arrays on first pass or if channels change
                    if (sumX == null || channels != ch)
                    {
                        channels = ch;
                        sumX = new double[ch];
                        sumX2 = new double[ch];
                        history = new float[ch, WindowSize];
                        historyIndex = 0;
                        sampleCount = 0;
                    }

                    if (outputMat == null || outputMat.Rows != ch || outputMat.Cols != nIn)
                        outputMat = new Mat(ch, nIn, Depth.F32, 1);

                    // Input depth: F32 (legacy, after ConvertScale) or U16/S16 straight from the headstage
                    // (1.1.76: no ConvertScale needed; the z-score is invariant to the ADC offset and scale).
                    Depth inDepth = input.Depth;
                    if (inDepth != Depth.F32 && inDepth != Depth.U16 && inDepth != Depth.S16)
                        throw new InvalidOperationException("SlidingWindowZScore: input depth must be F32, U16 or S16, got " + inDepth);
                    int inElem = inDepth == Depth.F32 ? sizeof(float) : sizeof(ushort);

                    unsafe
                    {
                        byte* inBytes = (byte*)input.Data.ToPointer();
                        byte* outBytes = (byte*)outputMat.Data.ToPointer();
                        int inStep = input.Step;
                        int outStep = outputMat.Step;

                        for (int t = 0; t < nIn; t++)
                        {
                            sampleCount++;
                            // L expands from 1 up to WindowSize at the start of the recording
                            double L = Math.Min(sampleCount, WindowSize);

                            for (int c = 0; c < ch; c++)
                            {
                                byte* src = inBytes + c * inStep + t * inElem;
                                float val = inDepth == Depth.F32 ? *(float*)src
                                          : inDepth == Depth.U16 ? (float)*(ushort*)src
                                          : (float)*(short*)src;
                                double valD = val; // Promote to float64 for math

                                // If buffer is full, subtract the oldest value falling out of the window
                                if (sampleCount > WindowSize)
                                {
                                    float oldVal = history[c, historyIndex];
                                    sumX[c] -= oldVal;
                                    sumX2[c] -= (double)oldVal * oldVal;
                                }

                                // Overwrite history buffer with new value and add to running sums
                                history[c, historyIndex] = val;
                                sumX[c] += valD;
                                sumX2[c] += valD * valD;

                                // Calculate exact mean and variance 
                                // Var(X) = E[X^2] - (E[X])^2
                                double mu = sumX[c] / L;
                                double var = (sumX2[c] / L) - (mu * mu);

                                // Float precision bounds check (matches np.maximum(var, 0.0))
                                if (var < 0.0) var = 0.0;

                                double sig = Math.Sqrt(var);
                                double z = (valD - mu) / (sig + Eps);

                                // Write output
                                *(float*)(outBytes + c * outStep + t * sizeof(float)) = (float)z;
                            }

                            // Advance circular buffer index
                            historyIndex = (historyIndex + 1) % WindowSize;
                        }
                    }

                    return outputMat;
                });
            });
        }
    }
}
