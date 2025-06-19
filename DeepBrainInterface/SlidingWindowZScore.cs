using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Description("Sliding-window z-score over the last WindowSize samples (per channel). " +
                 "Keeps exactly WindowSize points and recomputes mean/variance in O(Channels) each step.")]
    public class SlidingWindowZScore: Transform<Mat, Mat>
    {
        // How many most recent samples to use for mean/variance
        private int _windowSize = 1250;
        [Description("Number of samples to keep in the sliding window.")]
        public int WindowSize
        {
            get => _windowSize;
            set
            {
                if (value < 1) throw new ArgumentException("WindowSize must be ≥ 1.");
                _windowSize = value;
            }
        }

        [Description("Number of channels (rows) in each incoming Mat.")]
        public int Channels { get; set; } = 8;

        // Ring buffer: buffer[channel, position], length = WindowSize
        private float[,] buffer;
        private int writeIndex = 0;
        private int samplesSeen = 0;

        // Per-channel sums and sums of squares for the active window
        private double[] sum;     // sum[i] = Σ buffer[i,*]
        private double[] sumSq;   // sumSq[i] = Σ (buffer[i,*]²)

        private bool initialized = false;


        public override IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                // 1) Validate shape: must be [Channels × 1]
                if (input.Rows != Channels) throw new ArgumentException(
                    $"SlidingWindowZScore: expected {Channels} rows, got {input.Rows}.");

                // 2) Convert to float32 if needed
                Mat processedInput = input;
                if (input.Depth != Depth.F32)
                {
                    processedInput = new Mat(input.Size, Depth.F32, input.Channels);
                    CV.Convert(input, processedInput);
                }

                // Must be exactly 1 column
                if (processedInput.Cols != 1) throw new ArgumentException(
                    $"SlidingWindowZScore expects exactly 1 column; got {processedInput.Cols}.");

                // 3) On first frame, allocate ring buffer + sums
                if (!initialized)
                {
                    buffer = new float[Channels, _windowSize];
                    sum = new double[Channels];
                    sumSq = new double[Channels];
                    writeIndex = 0;
                    samplesSeen = 0;
                    initialized = true;
                }

                // 4) Read new sample x[i] from that one column
                float[] x = new float[Channels];
                for (int i = 0; i < Channels; i++)
                {
                    x[i] = (float)processedInput.GetReal(i, 0);
                }

                // 5) Remove the “old” value (if buffer is full), add the new one
                int oldIdx = writeIndex;
                int count = Math.Min(samplesSeen, _windowSize);

                for (int i = 0; i < Channels; i++)
                {
                    float oldValue = (samplesSeen < _windowSize)
                                     ? 0f
                                     : buffer[i, oldIdx];

                    // subtract oldValue from sums
                    sum[i] -= oldValue;
                    sumSq[i] -= (double)oldValue * oldValue;

                    // write new into ring buffer
                    buffer[i, oldIdx] = x[i];

                    // add new to sums
                    sum[i] += x[i];
                    sumSq[i] += (double)x[i] * x[i];
                }

                // advance writeIndex and total count
                writeIndex = (writeIndex + 1) % _windowSize;
                samplesSeen++;

                // 6) Compute mean and variance over the “current window size” nₜ = min(samplesSeen, WindowSize)
                int n_t = Math.Min(samplesSeen, _windowSize);
                var zScore = new Mat(new Size(1, Channels), Depth.F32, 1);

                for (int i = 0; i < Channels; i++)
                {
                    double μ = sum[i] / n_t;
                    double avgSq = sumSq[i] / n_t;
                    double var = avgSq - μ * μ;
                    if (var < 0) var = 0;  // numeric safety
                    double σ = Math.Sqrt(var);
                    if (σ < 1e-10) σ = 1e-10;  // clamp

                    double z = ((double)x[i] - μ) / σ;
                    zScore.SetReal(i, 0, z);
                }

                return zScore;
            });
        }
    }
}
