
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Reactive.Linq;
using Bonsai;

namespace DeepBrainInterface
{
    public class EnhancedRippleStateMachine : Transform<float[], RippleStateInfo>
    {
        private readonly Stopwatch stateTimer = new Stopwatch();
        private Queue<float> probabilityHistory = new Queue<float>();
        private const int HistorySize = 5;
        private const int MaxStateDurationMs = 50;

        [Description("Minimum time in state before transition (ms)")]
        public int MinStateDurationMs { get; set; } = 20;

        [Description("Probability trend threshold for state changes")]
        public float TrendThreshold { get; set; } = 0.1f;

        public override IObservable<RippleStateInfo> Process(IObservable<float[]> source)
        {
            return source.Select(probs =>
            {
                var maxProb = probs.Max();
                UpdateProbabilityHistory(maxProb);

                var trend = CalculateProbabilityTrend();
                var nextState = DetermineNextState(maxProb, trend);

                if (ShouldResetState())
                {
                    ResetState();
                    nextState = RippleState.NoRipple;
                }

                return new RippleStateInfo
                {
                    State = nextState,
                    CurrentSkip = DetermineSkipRate(nextState, trend),
                    Duration = stateTimer.ElapsedMilliseconds,
                    Trend = trend
                };
            });
        }

        private bool ShouldResetState()
        {
            return stateTimer.ElapsedMilliseconds > MaxStateDurationMs &&
                   currentState == RippleState.DefiniteRipple;
        }

        private void UpdateProbabilityHistory(float prob)
        {
            if (probabilityHistory.Count >= HistorySize)
                probabilityHistory.Dequeue();
            probabilityHistory.Enqueue(prob);
        }

        private float CalculateProbabilityTrend()
        {
            if (probabilityHistory.Count < 2) return 0;
            return probabilityHistory.Last() - probabilityHistory.Average();
        }

        private RippleState DetermineNextState(float prob, float trend)
        {
            if (stateTimer.ElapsedMilliseconds < MinStateDurationMs)
                return currentState;

            var nextState = prob switch
            {
                var p when p > 0.8f => RippleState.DefiniteRipple,
                var p when p > 0.5f && trend > TrendThreshold => RippleState.PossibleRipple,
                var p when p < 0.3f => RippleState.NoRipple,
                _ => currentState
            };

            if (nextState != currentState)
                stateTimer.Restart();

            return nextState;
        }
    }
}