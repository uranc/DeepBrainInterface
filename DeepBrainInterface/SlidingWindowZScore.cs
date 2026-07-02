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
                                float val = *(float*)(inBytes + c * inStep + t * sizeof(float));
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

//using Bonsai;
//using OpenCV.Net;
//using System;
//using System.ComponentModel;
//using System.Reactive.Linq;

//namespace DeepBrainInterface
//{
//    [Combinator]
//    [Description("EMA Z-score with symmetric clipping. No history buffer — GC friendly.")]
//    [WorkflowElementCategory(ElementCategory.Transform)]
//    public class SlidingWindowZScore
//    {
//        [Description("EMA window size in samples (alpha = 2/(N+1)).")]
//        public int WindowSize { get; set; } = 1250;

//        [Description("Symmetric clip threshold (output clamped to ±Threshold).")]
//        public float Threshold { get; set; } = 8f;

//        public IObservable<Mat> Process(IObservable<Mat> source)
//        {
//            return Observable.Defer(() =>
//            {
//                int channels = 0;
//                double alpha = 0.0;
//                double[] mu = null;
//                double[] var = null;
//                Mat outputMat = null;

//                return source.Select(input =>
//                {
//                    int ch = input.Rows;
//                    int nIn = input.Cols;

//                    if (mu == null || channels != ch)
//                    {
//                        channels = ch;
//                        alpha = 2.0 / (WindowSize + 1);
//                        mu = new double[ch];
//                        var = new double[ch];
//                    }

//                    if (outputMat == null || outputMat.Rows != ch || outputMat.Cols != nIn)
//                        outputMat = new Mat(ch, nIn, Depth.F32, 1);

//                    float threshold = Threshold;

//                    unsafe
//                    {
//                        byte* inBytes = (byte*)input.Data.ToPointer();
//                        byte* outBytes = (byte*)outputMat.Data.ToPointer();
//                        int inStep = input.Step;
//                        int outStep = outputMat.Step;

//                        for (int t = 0; t < nIn; t++)
//                        {
//                            for (int c = 0; c < ch; c++)
//                            {
//                                float val = *(float*)(inBytes + c * inStep + t * sizeof(float));
//                                double diff = val - mu[c];
//                                mu[c] += alpha * diff;
//                                var[c] = (1.0 - alpha) * (var[c] + alpha * diff * diff);
//                                double sig = Math.Sqrt(var[c]);
//                                double z = (val - mu[c]) / (sig + 1e-8);

//                                // Symmetric clip
//                                if (z > threshold) z = threshold;
//                                else if (z < -threshold) z = -threshold;

//                                *(float*)(outBytes + c * outStep + t * sizeof(float)) = (float)z;
//                            }
//                        }
//                    }

//                    return outputMat;
//                });
//            });
//        }
//    }
//}