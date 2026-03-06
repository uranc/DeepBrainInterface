using Bonsai;
using System;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime;
using System.ComponentModel;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Prints Server GC status to the console once, without altering the data stream.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class PrintServerGC
    {
        public IObservable<TSource> Process<TSource>(IObservable<TSource> source)
        {
            bool printed = false;

            return source.Do(x =>
            {
                if (!printed)
                {
                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("SERVER GC ENABLED: " + GCSettings.IsServerGC);
                    Console.WriteLine("=========================================\n");
                    printed = true;
                }
            });
        }
    }
}