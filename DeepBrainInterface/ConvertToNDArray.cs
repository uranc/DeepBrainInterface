using Bonsai;
using System;
using System.ComponentModel;
using static Tensorflow.Binding;
using OpenCV.Net;
using Tensorflow.NumPy;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Convert Mat or float array to NDArray")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ConvertToNDArray
    {
        public IObservable<NDArray> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                var array = new float[input.Rows * input.Cols];
                Marshal.Copy(input.Data, array, 0, array.Length);
                return np.array(array);
            });
        }

        public IObservable<NDArray> Process(IObservable<float[]> source)
        {
            return source.Select(input => np.array(input));
        }
    }
}
