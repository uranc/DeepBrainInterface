using Bonsai;
using System;
using System.ComponentModel;
using OpenCV.Net;
using System.Reactive.Linq;
using Tensorflow;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Converts NDArray to OpenCV Mat format with optimized performance.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public unsafe class ConvertNDArrayToMat : Transform<NDArray, Mat>
    {
        public override IObservable<Mat> Process(IObservable<NDArray> source)
        {
            return source.Select(input =>
            {
                if (input == null || input.shape.Length < 2)
                    throw new ArgumentException("Invalid input array");

                var shape = input.shape;
                var mat = new Mat(shape[0], shape[1], Depth.F32, 1);
                
                // Get direct pointer to NDArray data
                IntPtr srcPtr = input.BufferToArray();
                if (srcPtr == IntPtr.Zero)
                    throw new InvalidOperationException("Failed to get array buffer");

                // Direct memory copy
                Buffer.MemoryCopy(
                    srcPtr.ToPointer(),
                    mat.Data.ToPointer(),
                    mat.Step * mat.Rows,
                    shape[0] * shape[1] * sizeof(float));

                return mat;
            });
        }
    }
}
