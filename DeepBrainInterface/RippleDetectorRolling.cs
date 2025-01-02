using Bonsai;
using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using System.Windows.Forms.Design;
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

        [Description("Number of channels for the input")]
        public Int64 nChannels { get; set; } = 8;

        [Description("Path to the TensorFlow model file")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\simple_frozen_graph.pb";

        private void InitializeModel()
        {
            if (session == null)
            {
                graph = new Tensorflow.Graph();
                graph.as_default();
                graph.Import(ModelPath);
                session = new Session(graph);
                inputOperation = graph.OperationByName("x");
                outputOperation = graph.OperationByName("Identity");
            }
        }

        public IObservable<float[]> Process(IObservable<NDArray> source)
        {
            InitializeModel();
        
            const int MaxFreshestBatches = 8; // Adjust for how many freshest batches to keep
            const int MaxConcurrentBatches = 4; // Adjust for desired parallelism
            const int CombinedBatchSize = 2; // Combine 2 batches into 1 model input
        
            var rollingQueue = new Queue<NDArray>();
        
            return source
                .Do(input =>
                {
                    lock (rollingQueue)
                    {
                        rollingQueue.Enqueue(input);
                        if (rollingQueue.Count > MaxFreshestBatches)
                        {
                            rollingQueue.Dequeue(); // Drop the oldest batch
                        }
                    }
                })
                .Where(_ =>
                {
                    lock (rollingQueue)
                    {
                        return rollingQueue.Count >= CombinedBatchSize; // Ensure we have enough batches to combine
                    }
                })
                .SelectMany(_ =>
                {
                    // Combine multiple batches into one
                    NDArray combinedBatch;
                    lock (rollingQueue)
                    {
                        var batchesToCombine = rollingQueue.Take(CombinedBatchSize).ToArray();
                        rollingQueue = new Queue<NDArray>(rollingQueue.Skip(CombinedBatchSize)); // Remove combined batches
                        combinedBatch = np.concatenate(batchesToCombine, axis: 0); // Combine along time axis
                    }
        
                    // Process combined batch asynchronously
                    return Observable.Start(() =>
                    {
                        var results = session.run(outputOperation.outputs[0],
                            new FeedItem(inputOperation.outputs[0], combinedBatch.reshape((1, CombinedBatchSize * nTimesteps, nChannels))));
                        return np.squeeze(results).ToArray<float>();
                    });
                })
                .Merge(concurrency: MaxConcurrentBatches) // Process up to MaxConcurrentBatches in parallel
                .Subscribe(result =>
                {
                    Console.WriteLine($"Processed combined batch with size {result.Length}");
                });
        }
    }
}
