using System;
using System.Reactive.Linq;
using OpenCvSharp;
using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using Tensorflow;
using Tensorflow.NumPy;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Converts a float array to Mat format OpenCV.Net")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public static class ConvertFromNDToMat
    {
        public static IObservable<Mat> ConvertFromNDToMat(this IObservable<float[]> source, int nTimesteps)
        {
            return source.Select(resultData =>
            {
                // Create Mat from the float array
                var mat = new Mat(nTimesteps, 9, MatType.CV_32F, 1);
                mat.SetData(resultData);

                return mat;
            });
        }
    }
}
