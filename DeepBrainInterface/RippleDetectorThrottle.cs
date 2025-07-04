﻿//using Bonsai;
//using System;
//using System.ComponentModel;
//using System.Drawing.Design;
//using System.Linq;
//using System.Reactive.Linq;
//using System.Windows.Forms.Design;
//using Tensorflow;
//using Tensorflow.NumPy;

//namespace DeepBrainInterface
//{
//    [Combinator]
//    [Description("Detect ripples using CNN model")]
//    [WorkflowElementCategory(ElementCategory.Transform)]
//    public class RippleDetectorThrottle
//    {
//        private static Session session;
//        private static Tensorflow.Graph graph;
//        private static Operation inputOperation;
//        private static Operation outputOperation;

//        [Description("Number of timesteps for the input")]
//        public Int64 nTimesteps { get; set; } = 50;

//        [Description("Number of channels for the input")]
//        public Int64 nChannels { get; set; } = 8;

//        [Description("Path to the TensorFlow model file")]
//        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
//        public string ModelPath { get; set; } = @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\simple_frozen_graph.pb";

//        private void InitializeModel()
//        {
//            if (session == null)
//            {
//                graph = new Tensorflow.Graph();
//                graph.as_default();
//                graph.Import(ModelPath);
//                session = new Session(graph);
//                inputOperation = graph.OperationByName("x");
//                outputOperation = graph.OperationByName("Identity");
//            }
//        }

//        public IObservable<float[]> Process(IObservable<NDArray> source)
//        {
//            InitializeModel();
        
//            return source
//                .Throttle(TimeSpan.FromMilliseconds(ProcessingTime)) // Only process the latest batch
//                .SelectMany(input =>
//                {
//                    var results = session.run(outputOperation.outputs[0],
//                        new FeedItem(inputOperation.outputs[0], input.reshape((1, nTimesteps, nChannels))));
//                    return Observable.Return(np.squeeze(results).ToArray<float>());
//                });
//        }
//    }
//}

