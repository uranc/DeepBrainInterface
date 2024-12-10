using Bonsai;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing.Printing;
using System.Linq;
using System.Reactive.Linq;
using Tensorflow;
using static Tensorflow.Binding;
using Tensorflow.NumPy;
using System.IO;
using OneOf.Types;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Detect with CNN")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorBackup
    {

        public Int64 nTimesteps { get; set; } = 50;
        //public IObservable<IObservable<float>> Process(IObservable<NDArray> source)
        public IObservable<float[]> Process(IObservable<NDArray> source)
        {
            var graph = Detector.Generate();
            var sess = new Session(graph);
            var input_operation = graph.OperationByName("x");
            var output_operation = graph.OperationByName("Identity");

            return source.SelectMany(input =>
            {
                // TODO: process the input object and return the result.

                //static Graph graph = Detector.Generate();
                //print(sess);
                //print(input.shape);
                //    print(source);
                //    byte[] data = new byte[128 * 8];
                //var nn = np.array(source);

                //data = source.ToArray();
                //var nn = np.array(source.ToArray());
                //var results = sess.run(output_operation.outputs[0],
                //        new FeedItem(input_operation.outputs[0], aa));
                var results = sess.run(output_operation.outputs[0],
                        new FeedItem(input_operation.outputs[0], input.reshape((1, nTimesteps, 8))));
                //print(results.shape);
                //throw new NotImplementedException();

                //return default(int);
                results = np.squeeze(results);
                //print(results);
                //print(results[49]);
                return Observable.Return(results[49].ToArray<float>());
                //var floatOut = float.Parse(results[49].ToString());
                //var floatOut = float.Parse("0".ToString());
                //IObservable<float> floatOutObs = Observable.Return(floatOut);
                //return floatOutObs;
                //var argsort = /*results.ToString();*/
                //return Observable.Return(floatOut);
                //floatOut;
                //return source;
                //return input.reshape((1, 128, 8)();
            });
            //.Buffer(1)
            //.Select(list => list.ToArray());
        }
    }
}
