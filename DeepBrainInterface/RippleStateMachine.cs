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

    public class RippleStateInfo
    {
        public RippleState State { get; set; }
        public int CurrentSkip { get; set; }
        public int ActiveChannels { get; set; }
    }

    [Description("Tracks ripple states across multiple channels and controls skip logic for downstream nodes.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleStateMachine : Transform<IList<float>, RippleStateInfo>
    {
        private RippleState currentState = RippleState.NoRipple;

        [Description("Threshold for ripple detection on channels.")]
        public float DetectionThreshold { get; set; } = 0.7f;

        [Description("Number of channels required to exceed threshold.")]
        public int RequiredChannels { get; set; } = 1;

        [Description("Number of consecutive samples required to enter DefiniteRipple state.")]
        public int ConsecutiveSamplesRequired { get; set; } = 3;

        [Description("Skip rate for NoRipple state.")]
        public int NoRippleSkip { get; set; } = 10;

        [Description("Skip rate for PossibleRipple state.")]
        public int PossibleRippleSkip { get; set; } = 5;

        [Description("Skip rate for DefiniteRipple state.")]
        public int DefiniteRippleSkip { get; set; } = 1;

        private int consecutiveCount = 0;
        private const float ExitThresholdMultiplier = 0.8f; // Hysteresis
        private DateTime lastStateChange = DateTime.MinValue;
        private const int MinStateChangeDurationMs = 50;

        public override IObservable<RippleStateInfo> Process(IObservable<IList<float>> source)
        {
            return source.Select(inputs =>
            {
                int activeChannels = inputs.Count(v => v > DetectionThreshold);
                UpdateState(activeChannels);

                return new RippleStateInfo
                {
                    State = currentState,
                    CurrentSkip = GetCurrentSkip(),
                    ActiveChannels = activeChannels
                };
            });
        }

        private void UpdateState(int activeChannels)
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
                        lastStateChange = now;
                    }
                    break;

                case RippleState.PossibleRipple:
                    if (activeChannels >= RequiredChannels)
                    {
                        consecutiveCount++;
                        if (consecutiveCount >= ConsecutiveSamplesRequired) currentState = RippleState.DefiniteRipple;
                    }
                    else
                    {
                        currentState = RippleState.NoRipple;
                        consecutiveCount = 0;
                    }
                    break;

                case RippleState.DefiniteRipple:
                    if (activeChannels < exitThreshold)
                    {
                        currentState = RippleState.PossibleRipple;
                        lastStateChange = now;
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
