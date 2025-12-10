using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    // ==============================================================================
    // 1. DATA STRUCTURES
    // ==============================================================================
    public enum RippleState
    {
        NoRipple,
        Possible,
        Ripple
    }

    public class RippleOut
    {
        public RippleState State { get; set; }
        public float SignalProb { get; set; }
        public float ArtifactProb { get; set; }
        public bool IsArtifact { get; set; }
        public bool Trigger { get; set; }      // True on Rising Edge
        public bool GateOpen { get; set; }

        // Data Pass-through (Optional)
        public int StrideUsed { get; set; }
        public Mat SignalData { get; set; }    // Holds the raw 'sig' Mat
    }

    // ==============================================================================
    // 2. STATE MACHINE NODE
    // ==============================================================================
    [Combinator]
    [Description("Schmitt Trigger State Machine. Supports 4 Overloads.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleStateMachineMatBool
    {
        // ---- Thresholds ----
        [Category("Thresholds")]
        [Description("Signal Probability > High -> Start Ripple")]
        public float ThresholdHigh { get; set; } = 0.8f;

        [Category("Thresholds")]
        [Description("Signal Probability < Low -> End Ripple")]
        public float ThresholdLow { get; set; } = 0.3f;

        [Category("Thresholds")]
        [Description("Artifact Probability > Threshold -> IGNORE Signal")]
        public float ArtifactThreshold { get; set; } = 0.5f;

        // ---- Timing ----
        [Category("Timing")]
        [Description("Refractory period (samples) after ripple ends.")]
        public int RefractorySamples { get; set; } = 50;

        // ---- Internal State ----
        private bool _active;
        private int _refractoryCounter;

        // ==============================================================================
        // CORE LOGIC (Public for External Calls)
        // ==============================================================================
        public RippleOut Update(float signal, float artifact, bool gateOpen, Mat signalData = null)
        {
            var outState = new RippleOut
            {
                SignalProb = signal,
                ArtifactProb = artifact,
                GateOpen = gateOpen,
                IsArtifact = artifact > ArtifactThreshold,
                StrideUsed = 0,
                SignalData = signalData // <--- Carries the raw Mat
            };

            // 1. Gate & Artifact Check
            if (!gateOpen || outState.IsArtifact)
            {
                _active = false;
                _refractoryCounter = 0;
                outState.State = RippleState.NoRipple;
                return outState;
            }

            // 2. Refractory Check
            if (_refractoryCounter > 0)
            {
                _refractoryCounter--;
                _active = false;
                outState.State = RippleState.NoRipple;
                return outState;
            }

            // 3. Schmitt Trigger
            if (_active)
            {
                // In Ripple -> Check if we drop below Low
                if (signal < ThresholdLow)
                {
                    _active = false;
                    _refractoryCounter = RefractorySamples;
                    outState.State = RippleState.NoRipple;
                }
                else
                {
                    outState.State = RippleState.Ripple;
                }
            }
            else
            {
                // No Ripple -> Check if we rise above High
                if (signal > ThresholdHigh)
                {
                    _active = true;
                    outState.State = RippleState.Ripple;
                    outState.Trigger = true; // Rising Edge
                }
                else if (signal > ThresholdLow)
                {
                    outState.State = RippleState.Possible;
                }
                else
                {
                    outState.State = RippleState.NoRipple;
                }
            }

            return outState;
        }

        // Helper: Extract Signal/Artifact from a single Mat
        private void ParseMat(Mat m, out float signal, out float artifact)
        {
            signal = 0; artifact = 0;
            if (m == null) return;
            unsafe
            {
                float* ptr = (float*)m.Data.ToPointer();
                signal = ptr[0];

                // If Mat has >1 element, Index 1 is Artifact
                if ((m.Rows * m.Cols * m.Channels) > 1)
                    artifact = ptr[1];
            }
        }

        // ==============================================================================
        // THE 4 OVERLOADS
        // ==============================================================================

        // 1. Mat Only
        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            return source.Select(m =>
            {
                ParseMat(m, out float s, out float a);
                return Update(s, a, true, null);
            });
        }

        // 2. WithLatestFrom(Mat, Bool) -> Tuple<Mat, bool>
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
        {
            return source.Select(t =>
            {
                ParseMat(t.Item1, out float s, out float a);
                return Update(s, a, t.Item2, null);
            });
        }

        // 3. Tuple(Mat, Mat) -> (SignalMat, ArtifactMat)
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return source.Select(t =>
            {
                ParseMat(t.Item1, out float s, out _);
                ParseMat(t.Item2, out float a, out _);
                return Update(s, a, true, null);
            });
        }

        // 4. WithLatestFrom(Tuple(Mat, Mat), Bool) -> Tuple<Tuple<Mat, Mat>, bool>
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
        {
            return source.Select(t =>
            {
                var mats = t.Item1;
                ParseMat(mats.Item1, out float s, out _);
                ParseMat(mats.Item2, out float a, out _);
                return Update(s, a, t.Item2, null);
            });
        }
    }
}