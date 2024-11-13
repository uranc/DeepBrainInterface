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
    [Description("Detect with CNN")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetector
    {
        public Int64 nTimesteps { get; set; } = 50;

        //public IObservable<Mat> Process(IObservable<NDArray> source)
        public IObservable<float[]> Process(IObservable<NDArray> source)
        {
            // Load the TensorFlow graph
            var graph = Detector.Generate();
            var sess = new Session(graph);
            var inputOperation = graph.OperationByName("x");
            var outputOperation = graph.OperationByName("Identity");

            return source.SelectMany(input =>
            {
                // Run inference on the input NDArray
                var results = sess.run(outputOperation.outputs[0],
                    new FeedItem(inputOperation.outputs[0], input.reshape((1, nTimesteps, 8))));

                // Squeeze and reshape the results
                results = np.squeeze(results);
                //results = results.reshape((nTimesteps, 9));

                // Extract float array from NDArray
                var resultData = results.ToArray<float>();
                return Observable.Return(resultData);
            });
        }
    }
}
