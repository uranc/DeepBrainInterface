using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Prepends a single boolean value to the sequence")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class StartWithBool
    {
        public bool Value { get; set; } = true;

        public IObservable<bool> Process(IObservable<bool> source)
        {
            return source.StartWith(Value);
        }
    }
}