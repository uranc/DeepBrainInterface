using System;
using System.ComponentModel;
using System.Reactive.Linq;
using OpenCV.Net;
using Bonsai;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Downsamples each row (channel) in a matrix by averaging over non-overlapping column segments.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class DownsampleColumns
    {
        [Description("Downsampling factor (e.g., 12 for 30kHz to 2.5kHz).")]
        public int Factor { get; set; } = 12;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                // Check for null and valid data type
                if (input == null)
                    throw new InvalidOperationException("Input matrix cannot be null.");

                if (input.Depth != Depth.F32)
                    throw new InvalidOperationException($"Input matrix must be of type F32, but was {input.Depth}.");

                // Ensure factor is positive
                int adjustedFactor = Math.Max(1, Factor);

                // Calculate output dimensions
                int rows = input.Rows;
                int inputCols = input.Cols;
                int outputCols = inputCols / adjustedFactor;

                // Create reshaped matrices for efficient processing
                Mat result = new Mat(rows, outputCols, Depth.F32, 1);

                // Process each row (channel)
                // Process each row (channel)
                for (int r = 0; r < rows; r++)
                {
                    // Extract the current row
                    using (Mat rowMat = new Mat(new Size(inputCols, 1), Depth.F32, 1))
                    {
                        using (Mat subRect = input.GetSubRect(new Rect(0, r, inputCols, 1)))
                        {
                            CV.Copy(subRect, rowMat);
                        }
                        // Process the row using OpenCV functions
                        using (Mat reshapedRow = new Mat(adjustedFactor, outputCols, Depth.F32, 1))
                        {
                            // Reshape to prepare for averaging
                            for (int c = 0; c < outputCols; c++)
                            {
                                using (Mat segmentMat = rowMat.GetSubRect(new Rect(c * adjustedFactor, 0, adjustedFactor, 1)))
                                {
                                    // Calculate mean using OpenCV
                                    Scalar mean = CV.Avg(segmentMat);

                                    // Set output value
                                    result.SetReal(r, c, mean.Val0);
                                }
                            }
                        }
                    }
                }
                return result;
            });
        }
    }
}
