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
        // --- Detection state ---
        public RippleState State { get; set; }
        public bool   TTL          { get; set; }
        public float  Score        { get; set; }
        public int    EventCount   { get; set; }
        public float  Probability  { get; set; }
        public float  ArtifactProbability { get; set; }

        // --- Timing ---
        public ulong  Clock   { get; set; }  // hardware ulong clock of the freshest sample
        public int    Skipped { get; set; }  // 0 = inference ran normally; 1 = skipped (e.g. during refractory)

        // --- FSM config snapshot (populated by RippleStateMachineMatBool.Update / SuperNode) ---
        // Logged alongside detections so CSV captures the active settings at each event.
        public float  Threshold1   { get; set; }  // IgnoreBelow
        public float  Threshold2   { get; set; }  // WeakEvidence
        public float  Threshold3   { get; set; }  // StrongEvidence
        public float  EvidenceTarget { get; set; }
        public int    PostRippleMs { get; set; }
        public int    TriggerDelayMs { get; set; }
    }

    [Combinator]
    [Description("Accumulator FSM. Thread-safe with Leaky Bucket Grace Period.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class RippleStateMachineMatBool
    {
        [Category("General")] public bool DetectionEnabled { get; set; } = true;

        [Category("1. Thresholds (0.0 to 1.0)")]
        [Description("Threshold 1 (Floor): If probability drops below this, start the dip timer.")]
        public float Threshold1_IgnoreBelow { get; set; } = 0.10f;

        [Category("1. Thresholds (0.0 to 1.0)")]
        [Description("Threshold 2 (Weak): Awards 1 evidence point per frame.")]
        public float Threshold2_WeakEvidence { get; set; } = 0.50f;

        [Category("1. Thresholds (0.0 to 1.0)")]
        [Description("Threshold 3 (Strong): Awards 2 evidence points per frame.")]
        public float Threshold3_StrongEvidence { get; set; } = 0.80f;

        [Category("2. Scoring")]
        [Description("Total points needed to trigger the hardware TTL.")]
        public float TargetEvidenceScore { get; set; } = 5.0f;

        [Category("3. Leaky Bucket (Anti-Flicker)")]
        [Description("How many consecutive frames to WAIT during a dip BEFORE subtracting points.")]
        public int FramesToWaitBeforeDecay { get; set; } = 5;

        [Category("3. Leaky Bucket (Anti-Flicker)")]
        [Description("How many points to subtract from the score per frame AFTER the wait period.")]
        public float ScoreDecayPerFrame { get; set; } = 1.0f;

        [Category("4. TTL Output")] public int TriggerDelayMs { get; set; } = 0;
        [Category("4. TTL Output")] public bool RandomizeDelay { get; set; } = false;
        [Category("4. TTL Output")] public int PostRippleMs { get; set; } = 50;

        RippleState _state = RippleState.NoRipple;
        float _scoreTicks;
        int _ticksInPossible;
        int _ticksDipping;
        int _eventCount;

        bool _ttlArmed;
        long _ttlAtMs;
        bool _ttlHolding;
        long _ttlHoldUntilMs;

        static readonly Stopwatch Clock = Stopwatch.StartNew();
        private readonly Random _random = new Random();
        private readonly object _fsmLock = new object();

        public RippleOut Update(float signal, float artProb, bool gateOpen, Mat rawInput)
        {
            lock (_fsmLock)
            {
                long now = Clock.ElapsedMilliseconds;
                Mat triggerSnapshot = null;
                float reportingScore = _scoreTicks;

                // 1. Handle TTL Refractory Period
                if (_ttlHolding)
                {
                    if (now >= _ttlHoldUntilMs)
                    {
                        _ttlHolding = false;
                        _state = RippleState.NoRipple;
                        _scoreTicks = 0;
                        _ticksInPossible = 0;
                        _ticksDipping = 0;
                        reportingScore = 0f;
                    }
                    else return Pack(signal, artProb, 0f, true, null);
                }

                // 2. Handle Delayed TTL Trigger
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

                // 3. Master Gate Check
                bool allowed = DetectionEnabled && gateOpen;
                if (!allowed)
                {
                    _state = RippleState.NoRipple;
                    _scoreTicks = 0;
                    _ticksInPossible = 0;
                    _ticksDipping = 0;
                    return Pack(signal, artProb, 0f, false, null);
                }

                // 4. State Machine Logic
                switch (_state)
                {
                    case RippleState.NoRipple:
                        if (signal >= Threshold1_IgnoreBelow)
                        {
                            _state = RippleState.Possible;
                            _scoreTicks = 0;
                            _ticksInPossible = 0;
                            _ticksDipping = 0;
                        }
                        reportingScore = 0f;
                        break;

                    case RippleState.Possible:

                        if (signal >= Threshold1_IgnoreBelow)
                        {
                            // We are above the floor. Reset the dip timer and count frames.
                            _ticksDipping = 0;
                            _ticksInPossible++;

                            // Accumulate Evidence
                            if (signal >= Threshold3_StrongEvidence) _scoreTicks += 2.0f;
                            else if (signal >= Threshold2_WeakEvidence) _scoreTicks += 1.0f;

                            reportingScore = _scoreTicks;

                            // Check Trigger Condition
                            if (_scoreTicks >= TargetEvidenceScore)
                            {
                                _state = RippleState.Ripple;
                                _eventCount++;
                                float peakScore = _scoreTicks;
                                _scoreTicks = 0;
                                _ticksInPossible = 0;
                                _ticksDipping = 0;

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
                        }
                        else
                        {
                            // The signal dipped below the floor. Start the leaky bucket logic.
                            _ticksDipping++;

                            if (_ticksDipping > FramesToWaitBeforeDecay)
                            {
                                // We held our breath too long. Start bleeding points.
                                _scoreTicks -= ScoreDecayPerFrame;

                                if (_scoreTicks <= 0)
                                {
                                    // The bucket is empty. Hard reset.
                                    _state = RippleState.NoRipple;
                                    _scoreTicks = 0;
                                    _ticksInPossible = 0;
                                    _ticksDipping = 0;
                                    reportingScore = 0f;
                                }
                                else
                                {
                                    reportingScore = _scoreTicks;
                                }
                            }
                            else
                            {
                                // We are inside the grace period. Hold the score steady.
                                reportingScore = _scoreTicks;
                            }
                        }
                        break;

                    case RippleState.Ripple:
                        // After a successful trigger, wait until the event completely finishes
                        if (signal < Threshold1_IgnoreBelow)
                        {
                            _state = RippleState.NoRipple;
                            _scoreTicks = 0;
                            _ticksDipping = 0;
                            reportingScore = 0f;
                        }
                        break;
                }

                return Pack(signal, artProb, reportingScore, _ttlHolding, triggerSnapshot);
            }
        }

        private RippleOut Pack(float signal, float artProb, float currentScore, bool ttl, Mat data)
        {
            return new RippleOut
            {
                State               = _state,
                Score               = currentScore,
                Probability         = signal,
                ArtifactProbability = artProb,
                EventCount          = _eventCount,
                TTL                 = ttl,
                Skipped             = 0,   // standalone FSM never skips inference
                // Config snapshot — lets a single CSV Selector capture active settings at each event
                Threshold1          = Threshold1_IgnoreBelow,
                Threshold2          = Threshold2_WeakEvidence,
                Threshold3          = Threshold3_StrongEvidence,
                EvidenceTarget      = TargetEvidenceScore,
                PostRippleMs        = PostRippleMs,
                TriggerDelayMs      = TriggerDelayMs,
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

        // With clock — Zip(RippleDetectorCPU_output, clock) or Tuple<prob, clock> directly.
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, ulong>> source)
            => source.Select(t =>
            {
                var result = Update(GetVal(t.Item1), 0f, true, null);
                result.Clock = t.Item2;
                return result;
            });

        // With clock + BNO gate.
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, ulong>, bool>> source)
            => source.Select(t =>
            {
                var result = Update(GetVal(t.Item1.Item1), 0f, t.Item2, null);
                result.Clock = t.Item1.Item2;
                return result;
            });

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