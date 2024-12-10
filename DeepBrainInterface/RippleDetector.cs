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
    [Description("Detect ripples using CNN model")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetector
    {
        private static Session session;
        private static Tensorflow.Graph graph;
        private static Operation inputOperation;
        private static Operation outputOperation;

        [Description("Number of timesteps for the input")]
        public Int64 nTimesteps { get; set; } = 50;

        [Description("Path to the TensorFlow model file")]
        public string ModelPath { get; set; } = "default_model.pb";

        private void InitializeModel()
        {
            if (session == null)
            {
                //graph = Detector.Generate(ModelPath);
                var graph = new Tensorflow.Graph().as_default();
                graph.Import(ModelPath);
                session = new Session(graph);
                inputOperation = graph.OperationByName("x");
                outputOperation = graph.OperationByName("Identity");
            }
        }

        public IObservable<float[]> Process(IObservable<NDArray> source)
        {
            InitializeModel();
            
            return source.SelectMany(input =>
            {
                var results = session.run(outputOperation.outputs[0],
                    new FeedItem(inputOperation.outputs[0], input.reshape((1, nTimesteps, 8))));
                return Observable.Return(np.squeeze(results).ToArray<float>());
            });
        }
    }
}
