using Bonsai;
using System;
using System.ComponentModel;
using OpenCV.Net;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Converts a float array into an OpenCV.Net Mat.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ConvertNDArrayToMatFIX
    {
        public int Rows { get; set; } = 50;  // Number of rows in the Mat
        public int Cols { get; set; } = 8;   // Number of columns in the Mat

        public IObservable<Mat> Process(IObservable<float[]> source)
        {
            return source.Select(input =>
            {
                if (input.Length != Rows * Cols)
                {
                    throw new ArgumentException($"Input array must have exactly {Rows * Cols} elements.");
                }

                // Create a new Mat object
                var mat = new Mat(Rows, Cols, Depth.F32, 1);

                // Copy the data into the Mat object
                using (var matHeader = new Mat(Rows, Cols, Depth.F32, 1, input))
                {
                    CV.Copy(matHeader, mat);
                }

                return mat;
            });
        }
    }
}
