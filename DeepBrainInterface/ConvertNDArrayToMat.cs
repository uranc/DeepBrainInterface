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
    public class ConvertNDArrayToMat
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

                // Pin the array in memory
                var handle = System.Runtime.InteropServices.GCHandle.Alloc(input, System.Runtime.InteropServices.GCHandleType.Pinned);

                try
                {
                    // Create Mat directly from the pinned pointer
                    var mat = new Mat(Rows, Cols, Depth.F32, 1, handle.AddrOfPinnedObject());
                    return mat;
                }
                finally
                {
                    // Release the pinned handle
                    handle.Free();
                }
            });
        }
    }
}
