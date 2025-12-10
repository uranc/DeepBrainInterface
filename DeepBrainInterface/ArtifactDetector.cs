using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Simple Threshold: Returns True if Input Value (Index 0) > Threshold.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ArtifactDetector
    {
        [Range(0, 1)]
        [Editor(DesignTypes.SliderEditor, DesignTypes.UITypeEditor)]
        [Description("Threshold for detection.")]
        public float Threshold { get; set; } = 0.5f;

        public IObservable<bool> Process(IObservable<Mat> source)
        {
            return source.Select(m =>
            {
                if (m == null) return false;

                float val = 0f;
                // Just read the first float (Index 0)
                unsafe { val = *((float*)m.Data.ToPointer()); }

                return val > Threshold;
            });
        }
    }
}