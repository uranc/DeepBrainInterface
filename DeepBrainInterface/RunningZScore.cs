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
        private Mat runningM2;
        private int count;

        [Description("The window size for computing running statistics.")]
        public int WindowSize { get; set; } = 100;

        [Description("Number of channels in the input data.")]
        public int Channels { get; set; } = 8;

        public override IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                // Validate input dimensions
                if (input.Cols != Channels)
                {
                    throw new ArgumentException($"Input matrix must have {Channels} columns (channels). Got {input.Cols} instead.");
                }

                // Initialize statistics matrices if needed
                if (runningMean == null || runningMean.Cols != input.Cols)
                {
                    runningMean = Mat.Zeros(1, Channels, input.Depth);
                    runningM2 = Mat.Zeros(1, Channels, input.Depth);
                    count = 0;
                }

                count = Math.Min(count + 1, WindowSize);
                var scale = 1.0 / count;

                // Process each timepoint
                var output = new Mat(input.Size, input.Depth);
                for (int t = 0; t < input.Rows; t++)
                {
                    var currentRow = input.GetRow(t);
                    
                    // Update running statistics
                    var delta = new Mat();
                    CV.Sub(currentRow, runningMean, delta);
                    CV.ScaleAdd(delta, scale, runningMean, runningMean);

                    var delta2 = new Mat();
                    CV.Sub(currentRow, runningMean, delta2);
                    CV.Mul(delta, delta2, delta2);
                    CV.ScaleAdd(delta2, scale, runningM2, runningM2);

                    // Compute z-score for current timepoint
                    if (count > 1)
                    {
                        var std = new Mat();
                        CV.Sqrt(CV.Div(runningM2, count), std);
                        var zScore = new Mat();
                        CV.Sub(currentRow, runningMean, zScore);
                        CV.Div(zScore, std, zScore);
                        zScore.CopyTo(output.GetRow(t));
                    }
                }

                return output;
            });
        }
    }
}