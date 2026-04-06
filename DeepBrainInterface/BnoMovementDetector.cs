using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Description("Robust movement detector: Zero-allocation state machine for minimum latency.")]
    public class BnoMovementDetector : Transform<Vector3, BnoMovementState>
    {
        public float Alpha { get; set; } = 0.01f;
        public float Threshold { get; set; } = 0.2f;
        public int MedianWindow { get; set; } = 5;

        public override IObservable<BnoMovementState> Process(IObservable<Vector3> source)
        {
            // Defer creates the state EXACTLY ONCE when the workflow starts.
            // This prevents state leakage between runs without allocating per-frame.
            return Observable.Defer(() =>
            {
                // 1. Allocate buffers ONCE at startup
                var state = new DetectorState(MedianWindow);

                // 2. Process each frame
                return source.Select(input =>
                {
                    double norm = input.Length();

                    state.UpdateBuffer(norm);
                    double robustNorm = state.CalculateMedian();

                    state.AverageNorm = (state.AverageNorm == 0)
                        ? robustNorm
                        : (state.AverageNorm * (1.0 - Alpha)) + (robustNorm * Alpha);

                    // 3. Output as a Struct (Value Type = Zero Heap Allocation)
                    return new BnoMovementState
                    {
                        InstantNorm = norm,
                        RobustNorm = robustNorm,
                        AverageNorm = state.AverageNorm,
                        IsActive = state.AverageNorm > Threshold
                    };
                });
            });
        }

        private class DetectorState
        {
            public double AverageNorm;
            private readonly double[] _buffer;
            private readonly double[] _sortBuffer;
            private int _head;
            private int _count;

            public DetectorState(int size)
            {
                int safeSize = Math.Max(1, size);
                _buffer = new double[safeSize];
                _sortBuffer = new double[safeSize];
            }

            public void UpdateBuffer(double val)
            {
                _buffer[_head] = val;
                _head = (_head + 1) % _buffer.Length;
                if (_count < _buffer.Length) _count++;
            }

            public double CalculateMedian()
            {
                if (_count == 0) return 0;

                // Copy and sort only the valid data
                Array.Copy(_buffer, _sortBuffer, _count);
                Array.Sort(_sortBuffer, 0, _count);

                return _sortBuffer[_count / 2];
            }
        }
    }

    // Must remain a struct to avoid GC pressure!
    public struct BnoMovementState
    {
        public double InstantNorm;
        public double RobustNorm;
        public double AverageNorm;
        public bool IsActive;
    }
}