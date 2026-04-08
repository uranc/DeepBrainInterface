using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Zero-Allocation Sliding-window z-score. Uses unmanaged pointers. No GC Spikes.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class SlidingWindowZScore : IDisposable
    {
        [Description("Number of samples to keep in the sliding window.")]
        public int WindowSize { get; set; } = 1250;

        [Description("Number of channels (rows) in each incoming Mat.")]
        public int Channels { get; set; } = 8;

        private readonly object _lock = new object();
        private bool _initialized = false;

        private float[] _history;
        private double[] _sum;
        private double[] _sumSq;
        private int _writeIndex = 0;
        private int _samplesSeen = 0;

        private float[] _outBuffer;
        private GCHandle _hOutBuffer;
        private Mat _outMat;

        private void Initialize()
        {
            if (_initialized) return;

            _history = new float[Channels * WindowSize];
            _sum = new double[Channels];
            _sumSq = new double[Channels];

            _outBuffer = new float[Channels];
            _hOutBuffer = GCHandle.Alloc(_outBuffer, GCHandleType.Pinned);
            _outMat = new Mat(Channels, 1, Depth.F32, 1, _hOutBuffer.AddrOfPinnedObject());

            _writeIndex = 0;
            _samplesSeen = 0;
            _initialized = true;
        }

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                lock (_lock)
                {
                    if (!_initialized) Initialize();

                    if (input.Rows != Channels || input.Cols != 1 || input.Depth != Depth.F32)
                        throw new ArgumentException("Input must be a 32-bit float Mat with shape [Channels x 1]");

                    int count = Math.Min(_samplesSeen, WindowSize);

                    unsafe
                    {
                        float* src = (float*)input.Data.ToPointer();
                        float* dst = (float*)_hOutBuffer.AddrOfPinnedObject().ToPointer();

                        for (int i = 0; i < Channels; i++)
                        {
                            float newValue = src[i];
                            int historyOffset = (i * WindowSize) + _writeIndex;

                            float oldValue = (_samplesSeen < WindowSize) ? 0f : _history[historyOffset];
                            _sum[i] -= oldValue;
                            _sumSq[i] -= (double)oldValue * oldValue;

                            _history[historyOffset] = newValue;
                            _sum[i] += newValue;
                            _sumSq[i] += (double)newValue * newValue;

                            int n_t = Math.Max(1, count);
                            double mu = _sum[i] / n_t;
                            double avgSq = _sumSq[i] / n_t;
                            double variance = Math.Max(0, avgSq - (mu * mu));
                            double sigma = Math.Max(1e-10, Math.Sqrt(variance));

                            dst[i] = (float)((newValue - mu) / sigma);
                        }
                    }

                    _writeIndex = (_writeIndex + 1) % WindowSize;
                    _samplesSeen++;

                    return _outMat;
                }
            });
        }

        public void Dispose()
        {
            if (_hOutBuffer.IsAllocated) _hOutBuffer.Free();
            _initialized = false;
        }
    }
}