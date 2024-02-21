using Bonsai;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Transform1
    {
        public IObservable<bool> Process(IObservable<double> source)
        {
            return source.Select(input => input > 0);
        }
    }
}
