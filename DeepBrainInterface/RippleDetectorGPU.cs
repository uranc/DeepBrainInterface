using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design;
using System.Reactive.Linq;
using System.Runtime;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    public struct InferenceResult
    {
        public Mat Data;
        public double LatencyMs;
    }

    [Combinator]
    [Description("Latency-optimized ONNX inference node. IoBinding, pinned buffers, double-buffered output (no per-frame allocation).")]
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

        // Double-buffered output Mats. The pinned _bufOut is the live ORT target;
        // after each run we copy its single value into the inactive Mat and hand
        // that out, so an async consumer (save branch) always sees a stable buffer.
        private Mat _outMatA, _outMatB;
        private GCHandle _hOutA, _hOutB;
        private float[] _outA, _outB;
        private bool _useA;

        private volatile bool _isInitialized;
        private int _expectedInputBytes;
        private int _outputElementCount;
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
            GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;

            var opts = new SessionOptions { IntraOpNumThreads = 1, ExecutionMode = ExecutionMode.ORT_SEQUENTIAL };
            _session = new InferenceSession(ModelPath, opts);
            _binding = _session.CreateIoBinding();
            _runOpts = new RunOptions();

            int inputElementCount = BatchSize * TimePoints * Channels;
            _expectedInputBytes = inputElementCount * sizeof(float);
            _outputElementCount = BatchSize; // model outputs [BatchSize, 1]

            // Input: pinned, bound once.
            _bufIn = new float[inputElementCount];
            _hIn = GCHandle.Alloc(_bufIn, GCHandleType.Pinned);
            var valIn = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                new Memory<float>(_bufIn),
                new long[] { BatchSize, TimePoints, Channels });
            var inputName = System.Linq.Enumerable.First(_session.InputMetadata.Keys);
            _binding.BindInput(inputName, valIn);

            // Output: pinned live ORT target, bound once.
            _bufOut = new float[_outputElementCount];
            _hOut = GCHandle.Alloc(_bufOut, GCHandleType.Pinned);
            var valOut = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                new Memory<float>(_bufOut),
                new long[] { BatchSize, 1 });
            var outputName = System.Linq.Enumerable.First(_session.OutputMetadata.Keys);
            _binding.BindOutput(outputName, valOut);

            // Two stable output Mats over their own pinned buffers (double buffer).
            _outA = new float[_outputElementCount];
            _outB = new float[_outputElementCount];
            _hOutA = GCHandle.Alloc(_outA, GCHandleType.Pinned);
            _hOutB = GCHandle.Alloc(_outB, GCHandleType.Pinned);
            _outMatA = new Mat(BatchSize, 1, Depth.F32, 1, _hOutA.AddrOfPinnedObject());
            _outMatB = new Mat(BatchSize, 1, Depth.F32, 1, _hOutB.AddrOfPinnedObject());

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
                        Buffer.MemoryCopy(
                            m.Data.ToPointer(),
                            _hIn.AddrOfPinnedObject().ToPointer(),
                            _expectedInputBytes, _expectedInputBytes);
                    }

                    _timer.Restart();
                    _session.RunWithBinding(_runOpts, _binding);
                    _timer.Stop();

                    // Copy the live result into the inactive stable buffer, flip, hand it out.
                    // No heap allocation: both Mats and their buffers are pre-pinned.
                    var target = _useA ? _outA : _outB;
                    var targetMat = _useA ? _outMatA : _outMatB;
                    System.Buffer.BlockCopy(_bufOut, 0, target, 0, _outputElementCount * sizeof(float));
                    _useA = !_useA;

                    return new InferenceResult
                    {
                        Data = targetMat,
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
                if (_hOutA.IsAllocated) _hOutA.Free();
                if (_hOutB.IsAllocated) _hOutB.Free();
                _runOpts?.Dispose();
                _binding?.Dispose();
                _session?.Dispose();
                TimeEndPeriod(1);
            }
        }
    }
}