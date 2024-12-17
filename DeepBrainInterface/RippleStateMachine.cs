using Bonsai;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    public enum RippleState
    {
        NoRipple,
        PossibleRipple,
        DefiniteRipple
    }

    [Description("Tracks ripple states across multiple channels")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleStateMachine : Transform<IList<float>, RippleStateInfo>
    {
        private RippleState currentState = RippleState.NoRipple;
        private int eventCount = 0;
        private DateTime lastRippleTime = DateTime.MinValue;
        private float[] consecutiveThresholdCounts;
        private int channels = 1;

        [Description("Threshold values for each channel (comma-separated)")]
        public string DetectionThresholds { get; set; } = "0.7";

        [Description("Number of channels required to exceed threshold")]
        public int RequiredChannels { get; set; } = 1;

        [Description("Number of consecutive samples above threshold for definite ripple")]
        public int ConsecutiveSamplesRequired { get; set; } = 3;

        [Description("Buffer skip value for each state")]
        public int NoRippleSkip { get; set; } = 10;
        public int PossibleRippleSkip { get; set; } = 5;
        public int DefiniteRippleSkip { get; set; } = 1;

        private float[] thresholds;

        public override IObservable<RippleStateInfo> Process(IObservable<IList<float>> source)
        {
            return source.Select(inputs =>
            {
                // Lazy initialization of channel-specific settings
                if (consecutiveThresholdCounts == null || channels != inputs.Count)
                {
                    channels = inputs.Count;
                    consecutiveThresholdCounts = new float[channels];
                    InitializeThresholds();
                }

                UpdateState(inputs);
                return new RippleStateInfo
                {
                    State = currentState,
                    CurrentSkip = GetCurrentSkip(),
                    EventCount = eventCount,
                    TimeSinceLastRipple = DateTime.Now - lastRippleTime,
                    ModelOutputs = inputs.ToList(), // Store all channel values
                    ActiveChannels = CountActiveChannels(inputs)
                };
            });
        }

        private void InitializeThresholds()
        {
            var thresholdStrings = DetectionThresholds.Split(',');
            thresholds = new float[channels];

            // Fill thresholds array, repeating last value if needed
            for (int i = 0; i < channels; i++)
            {
                thresholds[i] = float.Parse(thresholdStrings[Math.Min(i, thresholdStrings.Length - 1)]);
            }
        }

        private void UpdateState(IList<float> inputs)
        {
            int activeChannels = CountActiveChannels(inputs);

            switch (currentState)
            {
                case RippleState.NoRipple:
                    if (activeChannels >= RequiredChannels)
                    {
                        currentState = RippleState.PossibleRipple;
                        ResetConsecutiveCounts();
                        UpdateConsecutiveCounts(inputs);
                    }
                    break;

                case RippleState.PossibleRipple:
                    if (activeChannels >= RequiredChannels)
                    {
                        UpdateConsecutiveCounts(inputs);
                        if (CheckConsecutiveCriteria())
                        {
                            currentState = RippleState.DefiniteRipple;
                            eventCount++;
                            lastRippleTime = DateTime.Now;
                        }
                    }
                    else
                    {
                        currentState = RippleState.NoRipple;
                        ResetConsecutiveCounts();
                    }
                    break;

                case RippleState.DefiniteRipple:
                    if (activeChannels < RequiredChannels)
                    {
                        currentState = RippleState.NoRipple;
                        ResetConsecutiveCounts();
                    }
                    break;
            }
        }

        private int CountActiveChannels(IList<float> inputs)
        {
            int count = 0;
            for (int i = 0; i < inputs.Count; i++)
            {
                if (inputs[i] >= thresholds[i])
                    count++;
            }
            return count;
        }

        private void UpdateConsecutiveCounts(IList<float> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                if (inputs[i] >= thresholds[i])
                    consecutiveThresholdCounts[i]++;
            }
        }

        private void ResetConsecutiveCounts()
        {
            Array.Clear(consecutiveThresholdCounts, 0, consecutiveThresholdCounts.Length);
        }

        private bool CheckConsecutiveCriteria()
        {
            return consecutiveThresholdCounts.Count(c => c >= ConsecutiveSamplesRequired) >= RequiredChannels;
        }

        private int GetCurrentSkip()
        {
            switch (currentState)
            {
                case RippleState.NoRipple:
                    return NoRippleSkip;
                case RippleState.PossibleRipple:
                    return PossibleRippleSkip;
                case RippleState.DefiniteRipple:
                    return DefiniteRippleSkip;
                default:
                    return NoRippleSkip;
            }
        }
    }

    public class RippleStateInfo
    {
        public RippleState State { get; set; }
        public int CurrentSkip { get; set; }
        public int EventCount { get; set; }
        public TimeSpan TimeSinceLastRipple { get; set; }
        public List<float> ModelOutputs { get; set; }
        public int ActiveChannels { get; set; }
    }
}