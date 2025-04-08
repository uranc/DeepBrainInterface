using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Implements a true circular buffer for Mat elements with fixed-size allocation.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class CircularBuffer
    {
        private Mat[] buffer;
        private int currentIndex;
        private bool bufferFilled;

        public CircularBuffer()
        {
            BufferSize = 50;
            buffer = new Mat[BufferSize];
            currentIndex = 0;
            bufferFilled = false;
        }

        [Description("The size of the circular buffer.")]
        public int BufferSize
        {
            get { return buffer?.Length ?? 0; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException(nameof(BufferSize), "Buffer size must be positive.");

                if (buffer == null || buffer.Length != value)
                {
                    // Clean up existing Mats if we're resizing
                    if (buffer != null)
                    {
                        foreach (var mat in buffer)
                        {
                            mat?.Dispose();
                        }
                    }

                    buffer = new Mat[value];
                    currentIndex = 0;
                    bufferFilled = false;
                }
            }
        }

        [Description("When true, emits the entire buffer once it's filled. When false, emits the buffer on every new input.")]
        public bool EmitOnlyWhenFilled { get; set; } = true;

        public IObservable<Mat[]> Process(IObservable<Mat> source)
        {
            return Observable.Create<Mat[]>(observer =>
            {
                return source.Subscribe(input =>
                {
                    // Clean up any existing Mat at the current position
                    buffer[currentIndex]?.Dispose();

                    // Store a clone of the input
                    buffer[currentIndex] = input.Clone();

                    // Move the index
                    currentIndex = (currentIndex + 1) % BufferSize;

                    // Check if we've filled the buffer at least once
                    if (currentIndex == 0)
                    {
                        bufferFilled = true;
                    }

                    // Decide whether to emit based on settings
                    if (!EmitOnlyWhenFilled || bufferFilled)
                    {
                        // Create a properly ordered array representing current buffer state
                        var result = new Mat[BufferSize];
                        
                        // If buffer is filled, start from the oldest element
                        int startIdx = bufferFilled ? currentIndex : 0;
                        
                        for (int i = 0; i < BufferSize; i++)
                        {
                            int idx = (startIdx + i) % BufferSize;
                            if (buffer[idx] != null)
                            {
                                result[i] = buffer[idx].Clone();
                            }
                        }
                        
                        observer.OnNext(result);
                    }
                },
                ex => observer.OnError(ex),
                () => 
                {
                    // Clean up on completion
                    foreach (var mat in buffer)
                    {
                        mat?.Dispose();
                    }
                    observer.OnCompleted();
                });
            });
        }
    }
}
