using Bonsai;
using System;
using System.ComponentModel;
using OpenCV.Net;
using System.Reactive.Linq;
using Tensorflow;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Converts NDArray to OpenCV Mat format with safety checks.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ConvertNDArrayToMat : Transform<NDArray, Mat>
    {
        public override IObservable<Mat> Process(IObservable<NDArray> source)
        {
            return source.Select(input =>
            {
                if (input == null)
                    throw new ArgumentNullException(nameof(input), "Input NDArray cannot be null");

                // Ensure input is in correct format
                if (input.dtype != TF_DataType.TF_FLOAT)
                {
                    input = input.cast(TF_DataType.TF_FLOAT);
                }

                var shape = input.shape;
                if (shape.Length < 2)
                    throw new ArgumentException("Input NDArray must have at least 2 dimensions");

                // Get data with bounds checking
                float[] data;
                try
                {
                    data = input.ToArray<float>();
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException("Failed to convert NDArray to float array", ex);
                }

                // Validate and clamp values
                for (int i = 0; i < data.Length; i++)
                {
                    if (float.IsNaN(data[i]) || float.IsInfinity(data[i]))
                        data[i] = 0f;
                    data[i] = Math.Max(0f, Math.Min(1f, data[i])); // Clamp between 0 and 1
                }

                // Create Mat with proper dimensions
                using (var mat = new Mat(shape[0], shape[1], Depth.F32, 1))
                {
                    Marshal.Copy(data, 0, mat.Data, data.Length);
                    return mat;
                }
            });
        }
    }
}
