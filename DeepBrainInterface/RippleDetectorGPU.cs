using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Runtime;
using System.Diagnostics;
using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;

// Alias to avoid "Process" naming collisions
using SysProcess = System.Diagnostics.Process;

namespace DeepBrainInterface
{
    // Result struct to allow plotting latency spikes
    public struct InferenceResult
    {
        public Mat Output;
        public double LatencyMicroseconds;
    }

    [Combinator]
    [Description("High-performance ONNX inference (Batch 1 or 2). Measures Latency.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorGPU : Combinator<Mat, InferenceResult>
    {
        // --- CONFIGURATION ---

        [Description("Path to the ONNX model file.")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", "System.Drawing.Design.UITypeEditor, System.Drawing")]
        [FileNameFilter("ONNX Models|*.onnx|All Files|*.*")]
        public string ModelPath { get; set; }

        [Description("Hardware backend.")]
        public InferenceProvider Provider { get; set; } = InferenceProvider.Cuda;

        [Description("Samples per channel (Time).")]
        public int TimePoint { get; set; } = 44;

        [Description("Number of channels.")]
        public int ChannelNo { get; set; } = 8;

        public enum InferenceProvider { Cpu, Cuda, TensorRt }

        // --- SINGLE INPUT (Batch = 1) ---
        public override IObservable<InferenceResult> Process(IObservable<Mat> source)
        {
            return Observable.Using(
                () => CreateEngine(batchSize: 1),
                (InferenceEngine engine) => source.Select(input => engine.Execute(input))
            );
        }

        // --- TUPLE INPUT (Batch = 2) ---
        public IObservable<InferenceResult> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return Observable.Using(
                () => CreateEngine(batchSize: 2),
                (InferenceEngine engine) => source.Select(input => engine.ExecuteBatch(input.Item1, input.Item2))
            );
        }

        // --- FACTORY ---
        private InferenceEngine CreateEngine(int batchSize)
        {
            // HARDCODED: Core 4, RealTime Priority
            OptimizeThread(coreIndex: 4);

            // HARDCODED: Output size is 1 per input (Scalar classification)
            int totalOutputSize = 1 * batchSize;

            return new InferenceEngine(ModelPath, Provider, batchSize, ChannelNo, TimePoint, totalOutputSize);
        }

        private static void OptimizeThread(int coreIndex)
        {
            try
            {
                if (Environment.OSVersion.Platform == PlatformID.Win32NT)
                {
                    SysProcess.GetCurrentProcess().ProcessorAffinity = new IntPtr(1 << coreIndex);
                    SysProcess.GetCurrentProcess().PriorityClass = ProcessPriorityClass.RealTime;
                    GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;
                }
            }
            catch { /* Fail silently or log if needed */ }
        }

        // --- ENGINE ---
        public class InferenceEngine : IDisposable
        {
            private InferenceSession Session;
            private OrtIoBinding IoBinding;
            private RunOptions RunOpts;

            private GCHandle InputPin, OutputPin;
            private float[] OutputBuffer;

            private int _time, _channels;
            private int _strideFloats;

            // Profiling
            private Stopwatch _watch;
            private double _freqInv;

            public InferenceEngine(string path, InferenceProvider provider, int batch, int channels, int time, int outSize)
            {
                _time = time;
                _channels = channels;
                _strideFloats = time * channels;
                OutputBuffer = new float[outSize];

                _watch = new Stopwatch();
                _freqInv = 1_000_000.0 / Stopwatch.Frequency; // Ticks -> Microseconds

                try
                {
                    var opts = new SessionOptions();
                    if (provider == InferenceProvider.Cuda)
                    {
                        try { opts.AppendExecutionProvider_CUDA(0); } catch { Console.WriteLine("[ERR] CUDA Failed"); }
                    }
                    else if (provider == InferenceProvider.TensorRt)
                    {
                        try { opts.AppendExecutionProvider_Tensorrt(0); opts.AppendExecutionProvider_CUDA(0); } catch { }
                    }
                    else
                    {
                        opts.IntraOpNumThreads = 1;
                        opts.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                    }

                    Session = new InferenceSession(path, opts);
                    RunOpts = new RunOptions();
                    IoBinding = Session.CreateIoBinding();

                    // Alloc Input Buffer [Batch * Time * Channel]
                    var inputBuffer = new float[batch * _strideFloats];

                    // Pin Memory
                    InputPin = GCHandle.Alloc(inputBuffer, GCHandleType.Pinned);
                    OutputPin = GCHandle.Alloc(OutputBuffer, GCHandleType.Pinned);

                    // Bind to ONNX Runtime
                    var mem = OrtMemoryInfo.DefaultInstance;

                    // Note: Check your model input shape. TCNs are often [Batch, Channels, Time] or [Batch, Time, Channels]
                    // The line below assumes [Batch, Time, Channels] based on your transpose logic.
                    // If your model is [Batch, Channels, Time], swap 'time' and 'channels' in the array below.
                    var inOrt = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(inputBuffer), new long[] { batch, time, channels });

                    var outOrt = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(OutputBuffer), new long[] { batch, outSize / batch });

                    IoBinding.BindInput(Session.InputMetadata.Keys.First(), inOrt);
                    IoBinding.BindOutput(Session.OutputMetadata.Keys.First(), outOrt);

                    inOrt.Dispose(); outOrt.Dispose();
                }
                catch (Exception ex) { Console.WriteLine($"[FATAL] {ex.Message}"); throw; }
            }

            public InferenceResult Execute(Mat m1)
            {
                _watch.Restart();

                unsafe
                {
                    float* ptr = (float*)InputPin.AddrOfPinnedObject();
                    TransposeToBuffer(m1, ptr);
                }

                var resMat = Run();
                _watch.Stop();

                return new InferenceResult { Output = resMat, LatencyMicroseconds = _watch.ElapsedTicks * _freqInv };
            }

            public InferenceResult ExecuteBatch(Mat m1, Mat m2)
            {
                _watch.Restart();

                unsafe
                {
                    float* ptr = (float*)InputPin.AddrOfPinnedObject();
                    TransposeToBuffer(m1, ptr);
                    TransposeToBuffer(m2, ptr + _strideFloats);
                }

                var resMat = Run();
                _watch.Stop();

                return new InferenceResult { Output = resMat, LatencyMicroseconds = _watch.ElapsedTicks * _freqInv };
            }

            // Optimized Transpose: [Channels, Time] -> [Time, Channels]
            private unsafe void TransposeToBuffer(Mat src, float* dstPtr)
            {
                float* srcPtr = (float*)src.Data.ToPointer();
                int srcStep = src.Step / sizeof(float); // Stride in floats

                int tMax = _time;
                int cMax = _channels;

                int dstIdx = 0;
                // Outer loop Time, Inner loop Channels = [Time, Channels] layout
                for (int t = 0; t < tMax; t++)
                {
                    for (int c = 0; c < cMax; c++)
                    {
                        // src is [Channels, Time], so index is (c * step) + t
                        dstPtr[dstIdx++] = srcPtr[(c * srcStep) + t];
                    }
                }
            }

            private Mat Run()
            {
                // Zero-copy execution via IoBinding
                Session.RunWithBinding(RunOpts, IoBinding);
                return Mat.FromArray(OutputBuffer);
            }

            public void Dispose()
            {
                Session?.Dispose(); IoBinding?.Dispose(); RunOpts?.Dispose();
                if (InputPin.IsAllocated) InputPin.Free();
                if (OutputPin.IsAllocated) OutputPin.Free();
            }
        }
    }
}