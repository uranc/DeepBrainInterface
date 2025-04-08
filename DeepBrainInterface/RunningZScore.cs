using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Description("Computes running z-score of multi-channel time series data.")]
    public class RunningZScore : Transform<Mat, Mat>
    {
        private Mat runningMean;
        private Mat runningVariance;
        private double forgettingFactor;

        [Description("The window size for computing running statistics.")]
        public int WindowSize
        {
            get { return _windowSize; }
            set
            {
                _windowSize = value;
                forgettingFactor = 2.0 / (_windowSize + 1); // Exponential moving average factor
            }
        }
        private int _windowSize = 50;

        [Description("Number of channels in the input data.")]
        public int Channels { get; set; } = 8;

        public RunningZScore()
        {
            forgettingFactor = 2.0 / (WindowSize + 1);
        }

        public override IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                // Validate input dimensions
                if (input.Rows != Channels)
                {
                    throw new ArgumentException($"Input matrix must have {Channels} rows (channels). Got {input.Rows} instead.");
                }

                // Convert input to F32 if needed
                Mat processedInput = input;
                if (input.Depth != Depth.F32 && input.Depth != Depth.F64)
                {
                    processedInput = new Mat(input.Size, Depth.F32, input.Channels);
                    CV.Convert(input, processedInput);
                }

                // Initialize statistics matrices if needed
                if (runningMean == null || runningMean.Rows != Channels || runningMean.Cols != processedInput.Cols)
                {
                    runningMean = new Mat(processedInput.Size, processedInput.Depth, processedInput.Channels);
                    CV.Copy(processedInput, runningMean);

                    // Initialize variance with small values to avoid division by zero
                    runningVariance = new Mat(processedInput.Size, processedInput.Depth, processedInput.Channels);
                    runningVariance.Set(Scalar.All(1e-4));
                    return new Mat(processedInput.Size, processedInput.Depth, processedInput.Channels); // Return zeros for first sample
                }

                // Compute delta (difference from mean)
                Mat delta = new Mat(processedInput.Size, processedInput.Depth, processedInput.Channels);
                CV.Sub(processedInput, runningMean, delta);

                // Update running mean 
                CV.ScaleAdd(delta, new Scalar(forgettingFactor), runningMean, runningMean);

                // Update running variance: var = (1-ff) * var + ff * delta²
                Mat deltaSquared = new Mat(processedInput.Size, processedInput.Depth, processedInput.Channels);
                CV.Mul(delta, delta, deltaSquared);
                CV.ConvertScale(runningVariance, runningVariance, 1.0 - forgettingFactor);
                CV.ScaleAdd(deltaSquared, new Scalar(forgettingFactor), runningVariance, runningVariance);

                // Compute standard deviation
                Mat stdDev = new Mat(processedInput.Size, processedInput.Depth, processedInput.Channels);
                CV.Pow(runningVariance, stdDev, 0.5);

                // Ensure minimum std dev to avoid division by zero
                for (int i = 0; i < stdDev.Rows; i++)
                {
                    for (int j = 0; j < stdDev.Cols; j++)
                    {
                        if (stdDev.GetReal(i, j) < 1e-10)
                        {
                            stdDev.SetReal(i, j, 1e-10);
                        }
                    }
                }

                // Calculate z-score
                Mat zScore = new Mat(processedInput.Size, processedInput.Depth, processedInput.Channels);
                CV.Div(delta, stdDev, zScore);

                // Clean up
                delta.Dispose();
                deltaSquared.Dispose();
                stdDev.Dispose();

                return zScore;
            });
        }
    }
}
