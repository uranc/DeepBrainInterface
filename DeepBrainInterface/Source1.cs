using Bonsai;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Description("")]
    [Combinator(MethodName = nameof(Generate))]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class Source1
    {
        public IObservable<int> Generate()
        {
            return Observable.Return(0);
        }
    }
}
