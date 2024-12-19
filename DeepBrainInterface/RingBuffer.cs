using Bonsai;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Reactive.Subjects;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Buffers a specified number of Mat elements and outputs them as a single Mat.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RingBuffer
    {
        private readonly Queue<Mat> bufferQueue;

        public RingBuffer()
        {
            bufferQueue = new Queue<Mat>();
        }

        [Description("The size of the buffer.")]
        public int BufferSize { get; set; } = 50;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Create<Mat>(observer =>
            {
                return source.Subscribe(input =>
                {
                    if (bufferQueue.Count == BufferSize)
                    {
                        bufferQueue.Dequeue();
                    }
                    bufferQueue.Enqueue(input);

                    if (bufferQueue.Count == BufferSize)
                    {
                        // Concatenate all Mats horizontally into a single Mat
                        Mat[] mats = bufferQueue.ToArray();
                        Mat concatenated = new Mat();
                        CV.HConcat(mats, concatenated);
                        observer.OnNext(concatenated);
                    }
                },
                ex => observer.OnError(ex),
                () => observer.OnCompleted());
            });
        }
    }
}
