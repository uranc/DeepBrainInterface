using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    public struct InferenceResult
    {
        public Mat Data;
        public double LatencyMs;
    }

    [Combinator]
    [Description("Raw ONNX Inference Node. Dynamic properties restored. Explicitly testing .Clone() allocation.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorGPU : IDisposable
    {
        [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod")] public static extern uint TimeBeginPeriod(uint uMilliseconds);
        [DllImport("winmm.dll", EntryPoint = "timeEndPeriod")] public static extern uint TimeEndPeriod(uint uMilliseconds);
        [DllImport("kernel32.dll")] static extern IntPtr GetCurrentThread();
        [DllImport("kernel32.dll")] static extern IntPtr SetThreadAffinityMask(IntPtr hThread, IntPtr dwThreadAffinityMask);

        private readonly object _inferenceLock = new object();
        private InferenceSession _session;
        private OrtIoBinding _binding;
        private RunOptions _runOpts;
        private float[] _bufIn, _bufOut;
        private GCHandle _hIn, _hOut;
        private Mat _outMat;
        private bool _isInitialized;
        private int _expectedInputBytes;
        private readonly Stopwatch _timer = new Stopwatch();

        [Category("1. Model Topology")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("1. Model Topology")]
        public int BatchSize { get; set; } = 1;

        [Category("1. Model Topology")]
        public int TimePoints { get; set; } = 44;

        [Category("1. Model Topology")]
        public int Channels { get; set; } = 8;

        [Category("2. Hardware Tuning")]
        public int TargetCore { get; set; } = 4;

        private void Initialize()
        {
            TimeBeginPeriod(1);

            var opts = new SessionOptions { IntraOpNumThreads = 1, ExecutionMode = ExecutionMode.ORT_SEQUENTIAL };
            _session = new InferenceSession(ModelPath, opts);
            _binding = _session.CreateIoBinding();
            _runOpts = new RunOptions();

            // Dynamically calculate memory requirements
            int inputElementCount = BatchSize * TimePoints * Channels;
            _expectedInputBytes = inputElementCount * sizeof(float);

            // 1. Setup Input Memory
            _bufIn = new float[inputElementCount];
            _hIn = GCHandle.Alloc(_bufIn, GCHandleType.Pinned);
            var valIn = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                new Memory<float>(_bufIn),
                new long[] { BatchSize, TimePoints, Channels }
            );

            var inputName = System.Linq.Enumerable.First(_session.InputMetadata.Keys);
            _binding.BindInput(inputName, valIn);

            // 2. Setup Output Memory
            _bufOut = new float[BatchSize];
            _hOut = GCHandle.Alloc(_bufOut, GCHandleType.Pinned);
            var valOut = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                new Memory<float>(_bufOut),
                new long[] { BatchSize, 1 }
            );

            var outputName = System.Linq.Enumerable.First(_session.OutputMetadata.Keys);
            _binding.BindOutput(outputName, valOut);

            _outMat = new Mat(BatchSize, 1, Depth.F32, 1, _hOut.AddrOfPinnedObject());

            SetThreadAffinityMask(GetCurrentThread(), (IntPtr)(1L << TargetCore));
            _isInitialized = true;
        }

        public IObservable<InferenceResult> Process(IObservable<Mat> source)
        {
            return source.Select(m =>
            {
                lock (_inferenceLock)
                {
                    if (!_isInitialized) Initialize();

                    unsafe
                    {
                        Buffer.MemoryCopy(m.Data.ToPointer(), _hIn.AddrOfPinnedObject().ToPointer(), _expectedInputBytes, _expectedInputBytes);
                    }

                    _timer.Restart();
                    _session.RunWithBinding(_runOpts, _binding);
                    _timer.Stop();

                    // BENCHMARKING .Clone()
                    return new InferenceResult
                    {
                        Data = _outMat.Clone(),
                        LatencyMs = _timer.Elapsed.TotalMilliseconds
                    };
                }
            });
        }

        public void Dispose()
        {
            lock (_inferenceLock)
            {
                if (_hIn.IsAllocated) _hIn.Free();
                if (_hOut.IsAllocated) _hOut.Free();
                _runOpts?.Dispose();
                _session?.Dispose();
                TimeEndPeriod(1);
            }
        }
    }
}