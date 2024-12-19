using Bonsai;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Reactive.Concurrency;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    public class EventInfo
    {
        public float Amplitude { get; set; }
        public DateTime TimeStamp { get; set; }
        public int TraceLength { get; set; }
        public float PeakValue { get; set; }
        public long Duration { get; set; }
    }

    [Description("Detects monotonically increasing traces above threshold.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class EventDetector : Transform<float, EventInfo>
    {
        private readonly Stopwatch eventTimer = new Stopwatch();
        private const int MaxTraceLength = 50; // 20ms at 2500Hz
        private float lastValue = 0;
        private int traceLength = 0;
        private float peakValue = 0;
        private bool inTrace = false;

        public float TraceThreshold { get; set; } = 0.5f;

        public override IObservable<EventInfo> Process(IObservable<float> source)
        {
            return source
                .ObserveOn(ThreadPoolScheduler.Instance)
                .Select(value =>
                {
                    if (!inTrace)
                    {
                        if (value > TraceThreshold)
                        {
                            eventTimer.Restart();
                            inTrace = true;
                            traceLength = 1;
                            peakValue = value;
                        }
                    }
                    else if (traceLength >= MaxTraceLength)
                    {
                        inTrace = false;
                        return CreateEventInfo();
                    }
                    else if (value > lastValue)
                    {
                        traceLength++;
                        peakValue = Math.Max(peakValue, value);
                    }
                    else
                    {
                        inTrace = false;
                        if (traceLength > 1)
                        {
                            var eventInfo = CreateEventInfo();
                            traceLength = 0;
                            return eventInfo;
                        }
                    }

                    lastValue = value;
                    return null;
                })
                .Where(info => info != null)
                .ObserveOn(ThreadPoolScheduler.Instance);
        }

        private EventInfo CreateEventInfo() =>
            new EventInfo
            {
                Amplitude = peakValue - TraceThreshold,
                TimeStamp = DateTime.UtcNow,
                TraceLength = traceLength,
                PeakValue = peakValue,
                Duration = eventTimer.ElapsedMilliseconds
            };
    }
}
