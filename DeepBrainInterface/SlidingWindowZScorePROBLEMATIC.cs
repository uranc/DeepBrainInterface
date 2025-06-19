using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Description("Streaming sliding-window z-score (8 ch × 1250 history, viewer-safe).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public unsafe class SlidingWindowZScorePROBLEMATIC : Transform<Mat, Mat>
    {
        [Description("Number of channels (rows).")]
        public int Channels { get; set; } = 8;

        [Description("Trailing window length (samples).")]
        public int WindowSize { get; set; } = 1250;

        /* ─── running statistics ─────────────────────────────────────────── */
        float[,] buf;      // [Channels × WindowSize] circular buffer
        double[] sum;      // Σx
        double[] sumSq;    // Σx²
        int head;     // index of oldest column

        /* ─── pool of output Mats (independent data buffers) ─────────────── */
        const int Pool = 32;
        Mat[] pool;
        int poolIdx;

        void AllocateState()
        {
            buf = new float[Channels, WindowSize];
            sum = new double[Channels];
            sumSq = new double[Channels];
            head = 0;

            pool = new Mat[Pool];
            for (int i = 0; i < Pool; i++)
                pool[i] = new Mat(Channels, 1, Depth.F32, 1); // each with its own data

            poolIdx = 0;
        }

        public override IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(sample =>
            {
                /* (re)allocate if first call or GUI changed params */
                if (buf == null ||
                    buf.GetLength(0) != Channels ||
                    buf.GetLength(1) != WindowSize)
                {
                    AllocateState();
                }

                /* pointer to incoming 8×1 float32 column */
                if (sample.Rows != Channels || sample.Cols != 1 || sample.Depth != Depth.F32)
                    throw new ArgumentException("Input Mat must be Channels×1, F32.");

                float* inPtr = (float*)sample.Data.ToPointer();

                /* choose next buffer from the pool and write into it */
                Mat outMat = pool[poolIdx];
                float* outPtr = (float*)outMat.Data.ToPointer();

                for (int c = 0; c < Channels; ++c)
                {
                    float xNew = inPtr[c];
                    float xOld = buf[c, head];
                    buf[c, head] = xNew;

                    sum[c] += xNew - xOld;
                    sumSq[c] += xNew * xNew - xOld * xOld;

                    double mean = sum[c] / WindowSize;
                    double var = (sumSq[c] - mean * mean * WindowSize) / WindowSize;
                    double std = Math.Sqrt(var > 1e-12 ? var : 1e-12);

                    outPtr[c] = (float)((xNew - mean) / std);
                }

                head = (head + 1) % WindowSize;
                poolIdx = (poolIdx + 1) % Pool;

                return outMat;                      // unique ref & unique data
            });
        }
    }
}
