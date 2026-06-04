using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace DeepBrainInterface
{
    public struct SuperNodeResult
    {
        public float Probability;
        public double LatencyMs;
        public int InferencesSkipped;
    }

    [Combinator]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorSuperNode : IDisposable
    {
        public string ModelPath { get; set; } = @"ripple_detector.onnx";
        public int TimePoints { get; set; } = 44;
        public int Channels { get; set; } = 8;

        private const int RingSize = 1024;
        private float[] _ring;
        private GCHandle _hRing;
        private volatile int _writeHead = 0;
        private CancellationTokenSource _cts;

        public IObservable<SuperNodeResult> Process(IObservable<Mat> source)
        {
            return Observable.Create<SuperNodeResult>(observer =>
            {
                _ring = new float[RingSize * Channels];
                _hRing = GCHandle.Alloc(_ring, GCHandleType.Pinned);
                _cts = new CancellationTokenSource();

                var sub = source.Subscribe(mat => {
                    int samples = mat.Cols;
                    unsafe
                    {
                        float* inPtr = (float*)mat.Data.ToPointer();
                        float* ringPtr = (float*)_hRing.AddrOfPinnedObject().ToPointer();
                        for (int t = 0; t < samples; t++)
                        {
                            int head = _writeHead;
                            for (int c = 0; c < Channels; c++) ringPtr[head * Channels + c] = inPtr[c * samples + t];
                            _writeHead = (head + 1) & (RingSize - 1);
                        }
                    }
                });

                Task.Factory.StartNew(() => RunPollingInference(observer, _cts.Token), TaskCreationOptions.LongRunning);

                return new CompositeDisposable(sub, Disposable.Create(() => _cts.Cancel()));
            });
        }

        private unsafe void RunPollingInference(IObserver<SuperNodeResult> observer, CancellationToken token)
        {
            var session = new InferenceSession(ModelPath);
            float[] buf = new float[TimePoints * Channels];
            int lastReadHead = 0;
            int skipped = 0;

            while (!token.IsCancellationRequested)
            {
                int head = _writeHead;
                if (((head - lastReadHead + RingSize) & (RingSize - 1)) < TimePoints)
                {
                    Thread.Sleep(1); continue;
                }

                if (((head - lastReadHead + RingSize) & (RingSize - 1)) > TimePoints * 2)
                {
                    skipped++;
                    lastReadHead = (head - TimePoints) & (RingSize - 1);
                }

                fixed (float* pRing = _ring, pBuf = buf)
                {
                    Buffer.MemoryCopy(pRing + (lastReadHead * Channels), pBuf, buf.Length * 4, TimePoints * Channels * 4);
                }

                var sw = Stopwatch.StartNew();
                var tensor = new DenseTensor<float>(buf, new int[] { 1, TimePoints, Channels });
                var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor(session.InputMetadata.Keys.First(), tensor) });
                float prob = results.First().AsTensor<float>().First();
                sw.Stop();

                observer.OnNext(new SuperNodeResult { Probability = prob, LatencyMs = sw.Elapsed.TotalMilliseconds, InferencesSkipped = skipped });
                lastReadHead = (lastReadHead + TimePoints) & (RingSize - 1);
            }
            session.Dispose();
        }

        public void Dispose() { _cts?.Cancel(); if (_hRing.IsAllocated) _hRing.Free(); }
    }
}