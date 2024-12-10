
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    public class EventInfo
    {
        public float Amplitude { get; set; }
        public DateTime TimeStamp { get; set; }
        public int TraceLength { get; set; }
        public float PeakValue { get; set; }
    }

    [Description("Detects monotonically increasing traces above threshold")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class EventDetector : Transform<float, EventInfo>
    {
        private float lastValue = 0;
        private int traceLength = 0;
        private float peakValue = 0;
        private bool inTrace = false;

        [Description("Minimum threshold to start trace detection")]
        public float TraceThreshold { get; set; } = 0.5f;

        [Description("Minimum increase required between samples")]
        public float MinimumIncrease { get; set; } = 0.01f;

        public override IObservable<EventInfo> Process(IObservable<float> source)
        {
            return source.Select(value =>
            {
                if (!inTrace && value > TraceThreshold && value > lastValue + MinimumIncrease)
                {
                    // Start new trace
                    inTrace = true;
                    traceLength = 1;
                    peakValue = value;
                }
                else if (inTrace)
                {
                    if (value > lastValue + MinimumIncrease)
                    {
                        // Continue trace
                        traceLength++;
                        peakValue = Math.Max(peakValue, value);
                    }
                    else
                    {
                        // End trace
                        inTrace = false;
                        if (traceLength > 1)
                        {
                            var eventInfo = new EventInfo
                            {
                                Amplitude = peakValue - TraceThreshold,
                                TimeStamp = DateTime.Now,
                                TraceLength = traceLength,
                                PeakValue = peakValue
                            };
                            traceLength = 0;
                            peakValue = 0;
                            lastValue = value;
                            return eventInfo;
                        }
                    }
                }

                lastValue = value;
                return null;
            })
            .Where(info => info != null);
        }
    }
}