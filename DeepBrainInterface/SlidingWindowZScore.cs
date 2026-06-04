using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Applies a running Z-Score using a pure mathematical counter. Strict F32 input. Zero ring buffers.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class SlidingWindowZScore
    {
        [Description("The history counter limit used to weight the running mean and variance. Defaults to 1250.")]
        public int WindowSize { get; set; } = 1250;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Defer(() =>
            {
                // Pure state trackers. No raw data is hoarded.
                float[] means = null;
                float[] variances = null;
                int count = 0;

                Mat outputMat = null;

                return source.Select(input =>
                {
                    int channels = input.Rows;
                    int incomingSamples = input.Cols;

                    // Initialize trackers exactly once
                    if (means == null)
                    {
                        means = new float[channels];
                        variances = new float[channels];
                        count = 0;
                    }

                    // Only reallocate the output matrix if the batch size physically changes
                    if (outputMat == null || outputMat.Cols != incomingSamples)
                    {
                        outputMat = new Mat(channels, incomingSamples, Depth.F32, 1);
                    }

                    unsafe
                    {
                        // Direct cast, trusting the upstream F32 contract
                        float* inData = (float*)input.Data.ToPointer();
                        float* outData = (float*)outputMat.Data.ToPointer();

                        // Step through timepoints chronologically to pool batches dynamically
                        for (int t = 0; t < incomingSamples; t++)
                        {
                            if (count < WindowSize) count++;
                            float alpha = 1.0f / count;
                            float oneMinusAlpha = 1.0f - alpha;

                            for (int c = 0; c < channels; c++)
                            {
                                int idx = c * incomingSamples + t;
                                float val = inData[idx];

                                // Exponential Moving Average (EMA) Update
                                float diff = val - means[c];
                                means[c] += alpha * diff;
                                variances[c] = oneMinusAlpha * (variances[c] + alpha * diff * diff);

                                float std = (float)Math.Sqrt(variances[c]);
                                if (std < 1e-6f) std = 1f; // Prevent division by zero flatlines

                                // Apply Z-Score directly
                                outData[idx] = (val - means[c]) / std;
                            }
                        }
                    }

                    // Clone kept purely to protect the upstream dsp:Buffer from overlapping memory writes
                    return outputMat.Clone();
                });
            });
        }
    }
}