//using Bonsai;
//using OpenCV.Net;
//using System;
//using System.ComponentModel;
//using System.Reactive.Linq;
//using Tensorflow;
//using Tensorflow.NumPy;

//namespace DeepBrainInterface
//{
//    /// <summary>
//    /// Converts a TensorFlow NDArray to an OpenCV Mat format.
//    /// </summary>
//    [Combinator]
//    [Description("Converts NDArray to OpenCV Mat format.")]
//    [WorkflowElementCategory(ElementCategory.Transform)]
//    public class ConvertNDArrayToMat : Transform<NDArray, Mat>
//    {
//        /// <summary>
//        /// The number of features per time point (e.g., 9).
//        /// </summary>
//        [Description("The number of features per time point.")]
//        public int FeatureCount { get; set; } = 9;

//        /// <summary>
//        /// Processes an observable sequence of NDArray objects and converts each one to an OpenCV Mat.
//        /// </summary>
//        /// <param name="source">The input sequence of NDArray objects.</param>
//        /// <returns>An observable sequence of Mat objects.</returns>
//        public override IObservable<Mat> Process(IObservable<NDArray> source)
//        {
//            return source.Select(input =>
//            {
//                // Validate that the input is not null
//                if (input == null)
//                    throw new ArgumentNullException(nameof(input));

//                // Check that the data type of the NDArray is float32
//                if (input.dtype != TF_DataType.TF_FLOAT)
//                    throw new NotSupportedException("Only float32 data type is supported.");

//                // Get total number of elements
//                int totalElements = (int)input.size;

//                // Calculate the number of rows (time points)
//                int numRows;

//                // Handle different shapes
//                if (input.ndim == 1)
//                {
//                    // Input is one-dimensional

//                    if (totalElements % FeatureCount != 0)
//                        throw new ArgumentException($"Input size must be a multiple of {FeatureCount}.", nameof(input));

//                    // Determine number of time points
//                    numRows = totalElements / FeatureCount;

//                    // Reshape the input to (numRows, FeatureCount)
//                    input = input.reshape((numRows, FeatureCount));
//                }
//                else if (input.ndim == 2)
//                {
//                    // Input is two-dimensional
//                    // Ensure the second dimension matches the feature count
//                    if (input.shape[1] != FeatureCount)
//                        throw new ArgumentException($"Second dimension size must be {FeatureCount}.", nameof(input));

//                    numRows = (int)input.shape[0];
//                }
//                else
//                {
//                    throw new ArgumentException("Input array must be one or two-dimensional.", nameof(input));
//                }

//                // Convert NDArray to a managed float array
//                float[] data = input.ToArray<float>();

//                // Create an OpenCV Mat with the specified dimensions and data type
//                var mat = new Mat(numRows, FeatureCount, Depth.F32, 1);

//                // Copy data from the float array to the Mat's unmanaged memory
//                System.Runtime.InteropServices.Marshal.Copy(data, 0, mat.Data, data.Length);

//                return mat;
//            });
//        }
//    }
//}
