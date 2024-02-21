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
        [Range(0.1, 2)]
        [Editor(DesignTypes.SliderEditor, DesignTypes.UITypeEditor)]
        public double PeriodSeconds { get; set; } = 0.5;
        public IObservable<double> Generate()
        {
            return Observable.Timer(dueTime: TimeSpan.Zero,
                period: TimeSpan.FromSeconds(PeriodSeconds))
                .Select(counter => Math.Sin(counter));
        }
    }
}
