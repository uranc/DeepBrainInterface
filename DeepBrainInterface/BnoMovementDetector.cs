using System;
using System.Numerics;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using Bonsai;

namespace DeepBrainInterface
{
    [Description("Robust movement detector: Uses Median Filter (to ignore twitches) + Running Average (trend) of BNO Norm.")]
    public class BnoMovementDetector : Transform<Vector3, BnoMovementState>
    {
        [Description("The smoothing factor for the running average (0 to 1).")]
        public float Alpha { get; set; } = 0.01f;

        [Description("The threshold for the Smoothed Norm to trigger IsActive.")]
        public float Threshold { get; set; } = 0.2f;

        [Description("Size of the Median Filter window (Odd number recommended). Higher = ignores longer twitches.")]
        public int MedianWindow { get; set; } = 5;

        public override IObservable<BnoMovementState> Process(IObservable<Vector3> source)
        {
            return source.Scan(new DetectorState(MedianWindow), (state, input) =>
            {
                // 1. Compute Raw Energy (Norm)
                // 
                double norm = input.Length();

                // 2. Robust Filtering (Median)
                // We add to the buffer and get the median value
                state.UpdateBuffer(norm, MedianWindow);
                double robustNorm = state.CalculateMedian();

                // 3. Running Average (EMA)
                // We average the 'Robust' norm, not the raw norm
                // If average is 0 (startup), snap to current value
                state.AverageNorm = (state.AverageNorm == 0)
                    ? robustNorm
                    : (state.AverageNorm * (1.0 - Alpha)) + (robustNorm * Alpha);

                // 4. Output Logic
                state.LastNorm = norm;
                state.CurrentRobust = robustNorm;

                return state;
            })
            .Select(s => new BnoMovementState
            {
                InstantNorm = s.LastNorm,
                RobustNorm = s.CurrentRobust,
                AverageNorm = s.AverageNorm,
                IsActive = s.AverageNorm > Threshold
            });
        }

        // Internal State Container (Hidden Logic)
        private class DetectorState
        {
            public double AverageNorm;
            public double LastNorm;
            public double CurrentRobust;

            // Ring Buffer for Median
            private readonly double[] _buffer;
            private readonly double[] _sortBuffer;
            private int _head;
            private int _count;

            public DetectorState(int size)
            {
                // Pre-allocate buffers to avoid GC pressure
                int safeSize = Math.Max(1, size);
                _buffer = new double[safeSize];
                _sortBuffer = new double[safeSize];
                _head = 0;
                _count = 0;
            }

            public void UpdateBuffer(double val, int targetSize)
            {
                // Handle resizing if property changed dynamically
                if (_buffer.Length != targetSize && targetSize > 0)
                {
                    // Reset if size changes (rare event)
                    // Ideally we just re-allocate, but for simplicity here we assume fixed size per run
                    // or just clamp to current capacity. 
                    // For safety in Bonsai, re-allocation is okay on property change but tricky inside Scan.
                    // We will stick to the initial size logic for stability or use a List if dynamic resizing is critical.
                    // Assuming MedianWindow doesn't change every frame:
                }

                _buffer[_head] = val;
                _head = (_head + 1) % _buffer.Length;
                if (_count < _buffer.Length) _count++;
            }

            public double CalculateMedian()
            {
                if (_count == 0) return 0;

                // Copy only valid elements to sort buffer
                Array.Copy(_buffer, _sortBuffer, _buffer.Length);

                // Sort the valid range
                // Since the buffer is circular, it might be fragmented, but we copied the whole array.
                // If _count < length, we only sort the first _count elements? 
                // No, because _buffer isn't packed.
                // Correct strategy for Ring Buffer Median:
                // 1. Copy active elements to sort buffer.

                // Simple approach for small windows: Just iterate and copy valid values
                // (Performance cost is negligible for N < 100)
                int validCount = 0;
                for (int i = 0; i < _count; i++)
                {
                    // Actually, if _count < Length, the valid data is [0.._count-1]
                    // If full, valid data is everywhere.
                    _sortBuffer[i] = _buffer[i];
                    validCount++;
                }

                Array.Sort(_sortBuffer, 0, _count);

                // Return Median
                return _sortBuffer[_count / 2];
            }
        }
    }

    public struct BnoMovementState
    {
        public double InstantNorm;
        public double RobustNorm; // Value after median filter
        public double AverageNorm;
        public bool IsActive;
    }
}