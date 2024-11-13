using Bonsai;
using System;
using System.ComponentModel;
using static Tensorflow.Binding;
using OpenCV.Net;
using Tensorflow.NumPy;
using System.Reactive.Linq;
using Google.Protobuf;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("DetectStuff")]
    [WorkflowElementCategory(ElementCategory.Transform)]

    public class ConvertToNDArray
    {
        public IObservable<NDArray> Process(IObservable<float[]> source)
        {
            var nn = source.Select(input => np.array(input));
            return nn;
        }
    }
}
