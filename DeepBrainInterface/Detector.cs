using Bonsai;
using System;
using System.ComponentModel;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("DBI")]
    [WorkflowElementCategory(ElementCategory.Transform)]

    public static class Detector
    {
        private static string modelPath = @"frozen_models\simple_frozen_graph.pb";

        public static Graph Generate()
        {
            Console.WriteLine("Loading Model");

            var graph = new Graph().as_default();
            graph.Import(modelPath);

            return graph;
        }
    }
}

