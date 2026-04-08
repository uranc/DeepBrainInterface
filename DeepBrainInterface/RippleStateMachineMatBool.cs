using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    public enum RippleState { NoRipple, Possible, Ripple }

    public struct RippleOut
    {
        public RippleState State { get; set; }
        public bool TTL { get; set; }
        public float Score { get; set; }
        public int EventCount { get; set; }
        public float Probability { get; set; }
        public Mat SignalData { get; set; }
        public float ArtifactProbability { get; set; }
        public int StrideUsed { get; set; }
    }

    [Combinator]
    [Description("Accumulator FSM. Inputs: Signal (Mat) + Master Gate (Bool).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class RippleStateMachineMatBool
    {
        [Category("General")] public bool DetectionEnabled { get; set; } = true;
        [Category("Thresholds")] public float GateThreshold { get; set; } = 0.10f;
        [Category("Thresholds")] public float EnterThreshold { get; set; } = 0.50f;
        [Category("Thresholds")] public float ConfirmThreshold { get; set; } = 0.80f;
        [Category("Thresholds")] public float EventScoreThreshold { get; set; } = 2.5f;
        [Category("Thresholds")] public int GraceSamples { get; set; } = 5;
        [Category("Thresholds")] public float GraceRate { get; set; } = 1.0f;
        [Category("TTL Output")] public int TriggerDelayMs { get; set; } = 0;
        [Category("TTL Output")] public bool RandomizeDelay { get; set; } = false;
        [Category("TTL Output")] public int PostRippleMs { get; set; } = 50;

        RippleState _state = RippleState.NoRipple;
        float _scoreTicks;
        int _ticksInPossible;
        int _eventCount;

        bool _ttlArmed;
        long _ttlAtMs;
        bool _ttlHolding;
        long _ttlHoldUntilMs;

        static readonly Stopwatch Clock = Stopwatch.StartNew();
        private readonly Random _random = new Random();

        public RippleOut Update(float signal, float artProb, bool gateOpen, Mat rawInput)
        {
            long now = Clock.ElapsedMilliseconds;
            Mat triggerSnapshot = null;
            float reportingScore = _scoreTicks;

            if (_ttlHolding)
            {
                if (now >= _ttlHoldUntilMs)
                {
                    _ttlHolding = false;
                    _state = RippleState.NoRipple;
                    _scoreTicks = 0;
                    _ticksInPossible = 0;
                    reportingScore = 0;
                }
                else return Pack(signal, artProb, 0f, true, null);
            }

            if (_ttlArmed)
            {
                if (now >= _ttlAtMs)
                {
                    _ttlArmed = false;
                    _ttlHolding = true;
                    _ttlHoldUntilMs = now + Math.Max(1, PostRippleMs);
                    triggerSnapshot = rawInput?.Clone();
                    return Pack(signal, artProb, 0f, true, triggerSnapshot);
                }
                return Pack(signal, artProb, reportingScore, false, null);
            }

            bool allowed = DetectionEnabled && gateOpen;
            if (!allowed)
            {
                _state = RippleState.NoRipple;
                _scoreTicks = 0;
                _ticksInPossible = 0;
                return Pack(signal, artProb, 0f, false, null);
            }

            float scoreTarget = EventScoreThreshold * 2.0f;

            switch (_state)
            {
                case RippleState.NoRipple:
                    if (signal >= GateThreshold)
                    {
                        _state = RippleState.Possible;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;
                    }
                    reportingScore = 0f;
                    break;

                case RippleState.Possible:
                    _ticksInPossible++;
                    if (signal >= ConfirmThreshold) _scoreTicks += 2.0f;
                    else if (signal >= EnterThreshold) _scoreTicks += 1.0f;

                    reportingScore = _scoreTicks;

                    if (_scoreTicks >= scoreTarget)
                    {
                        _state = RippleState.Ripple;
                        _eventCount++;
                        float peakScore = _scoreTicks;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;

                        if (TriggerDelayMs > 0)
                        {
                            _ttlArmed = true;
                            int actualDelay = RandomizeDelay ? _random.Next(0, TriggerDelayMs + 1) : TriggerDelayMs;
                            _ttlAtMs = now + actualDelay;
                            return Pack(signal, artProb, peakScore, false, null);
                        }
                        else
                        {
                            _ttlHolding = true;
                            _ttlHoldUntilMs = now + Math.Max(1, PostRippleMs);
                            triggerSnapshot = rawInput?.Clone();
                            return Pack(signal, artProb, peakScore, true, triggerSnapshot);
                        }
                    }
                    else if (signal < GateThreshold)
                    {
                        _state = RippleState.NoRipple;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;
                        reportingScore = 0f;
                    }
                    else if (_ticksInPossible > GraceSamples)
                    {
                        _scoreTicks -= GraceRate;
                        if (_scoreTicks < 0) _scoreTicks = 0;
                        reportingScore = _scoreTicks;
                    }
                    break;

                case RippleState.Ripple:
                    if (signal < GateThreshold)
                    {
                        _state = RippleState.NoRipple;
                        _scoreTicks = 0;
                        reportingScore = 0f;
                    }
                    break;
            }

            return Pack(signal, artProb, reportingScore, _ttlHolding, triggerSnapshot);
        }

        private RippleOut Pack(float signal, float artProb, float currentScore, bool ttl, Mat data)
        {
            return new RippleOut
            {
                State = _state,
                Score = currentScore * 0.5f,
                Probability = signal,
                ArtifactProbability = artProb,
                EventCount = _eventCount,
                TTL = ttl,
                SignalData = data,
                StrideUsed = 0
            };
        }

        private float GetVal(Mat m)
        {
            if (m == null) return 0f;
            unsafe { return *((float*)m.Data.ToPointer()); }
        }

        public IObservable<RippleOut> Process(IObservable<Mat> source)
            => source.Select(m => Update(GetVal(m), 0f, true, null));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
            => source.Select(t => Update(GetVal(t.Item1), 0f, t.Item2, null));

        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
        {
            return source.Select(t =>
            {
                float prob = GetVal(t.Item1.Item1);
                Mat rawData = t.Item1.Item2;
                bool gate = t.Item2;
                return Update(prob, 0f, gate, rawData);
            });
        }
    }
}