using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace DeepBrainInterface
{
    public struct InferenceResult
    {
        public bool IsValid { get; set; }
        public Mat Data { get; set; }
        public double LatencyMs { get; set; }
    }

    [Description("Hyper-Optimized CPU Inference. Timer Locked. Affinity Pinned.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorGPU : Transform<Mat, InferenceResult>, IDisposable
    {
        /* ───── Windows Kernel P/Invoke Hooks ───── */
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        const uint _MCW_DN = 0x03000000;
        const uint _DN_FLUSH = 0x01000000;

        [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod", SetLastError = true)]
        public static extern uint TimeBeginPeriod(uint uMilliseconds);

        [DllImport("winmm.dll", EntryPoint = "timeEndPeriod", SetLastError = true)]
        public static extern uint TimeEndPeriod(uint uMilliseconds);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool SetProcessWorkingSetSize(IntPtr hProcess, IntPtr dwMinimumWorkingSetSize, IntPtr dwMaximumWorkingSetSize);

        [DllImport("kernel32.dll")]
        static extern IntPtr GetCurrentThread();

        [DllImport("kernel32.dll")]
        static extern IntPtr SetThreadAffinityMask(IntPtr hThread, IntPtr dwThreadAffinityMask);

        /* ───── UI Properties ───── */
        [Category("1. Model")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("2. Settings")] public int IntraOpNumThreads { get; set; } = 1;
        [Category("3. Dimensions")] public int BatchSize { get; set; } = 1;
        [Category("3. Dimensions")] public int TimePoints { get; set; } = 44;
        [Category("3. Dimensions")] public int Channels { get; set; } = 8;
        [Category("4. Hardware Control")] public bool AllowSpinning { get; set; } = true;

        /* ───── State ───── */
        private readonly object _inferenceLock = new object();
        private InferenceSession _session;
        private OrtIoBinding _binding;
        private RunOptions _runOpts;
        private readonly Stopwatch _timer = new Stopwatch();
        private float[] _bufIn, _bufOut;
        private GCHandle _hIn, _hOut;
        private OrtValue _valIn, _valOut;
        private Mat _outMat;
        private int _inputStride;
        private int _expectedBytes;
        private bool _isInitialized = false;

        private void InitializeSession()
        {
            if (_isInitialized) return;

            // 1. Kernel Hooks
            try { TimeBeginPeriod(1); } catch { }
            try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { }
            try { SetProcessWorkingSetSize(System.Diagnostics.Process.GetCurrentProcess().Handle, (IntPtr)(-1), (IntPtr)(-1)); } catch { }
            try { System.Diagnostics.Process.GetCurrentProcess().PriorityClass = ProcessPriorityClass.RealTime; } catch { }

            _inputStride = TimePoints * Channels;
            _expectedBytes = _inputStride * sizeof(float);

            // 2. ONNX Config
            var opts = new SessionOptions
            {
                IntraOpNumThreads = IntraOpNumThreads,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                EnableCpuMemArena = false,
                EnableMemoryPattern = false
            };
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            opts.AddSessionConfigEntry("session.execution_mode", "ORT_SEQUENTIAL");
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", AllowSpinning ? "1" : "0");

            _session = new InferenceSession(ModelPath, opts);
            _binding = _session.CreateIoBinding();
            _runOpts = new RunOptions();
            opts.Dispose();

            // 3. Memory
            _bufIn = new float[BatchSize * _inputStride];
            _hIn = GCHandle.Alloc(_bufIn, GCHandleType.Pinned);
            _bufOut = new float[BatchSize];
            _hOut = GCHandle.Alloc(_bufOut, GCHandleType.Pinned);

            var memInfo = OrtMemoryInfo.DefaultInstance;
            _valIn = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufIn), new long[] { BatchSize, TimePoints, Channels });
            _valOut = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufOut), new long[] { BatchSize, 1 });

            _binding.BindInput(_session.InputMetadata.Keys.First(), _valIn);
            _binding.BindOutput(_session.OutputMetadata.Keys.First(), _valOut);

            _outMat = new Mat(BatchSize, 1, Depth.F32, 1, _hOut.AddrOfPinnedObject());
            _isInitialized = true;
        }

        public override IObservable<InferenceResult> Process(IObservable<Mat> source)
        {
            return source.Select(m =>
            {
                lock (_inferenceLock)
                {
                    if (!_isInitialized) InitializeSession();
                    SetThreadAffinityMask(GetCurrentThread(), (IntPtr)(1L << 0));

                    unsafe
                    {
                        float* dst = (float*)_hIn.AddrOfPinnedObject().ToPointer();
                        Buffer.MemoryCopy(m.Data.ToPointer(), dst, _expectedBytes, _expectedBytes);
                    }

                    _timer.Restart();
                    _session.RunWithBinding(_runOpts, _binding);
                    _timer.Stop();

                    return new InferenceResult { Data = _outMat, LatencyMs = _timer.Elapsed.TotalMilliseconds, IsValid = true };
                }
            });
        }

        public void Dispose()
        {
            lock (_inferenceLock)
            {
                if (_hIn.IsAllocated) _hIn.Free();
                if (_hOut.IsAllocated) _hOut.Free();
                _session?.Dispose();
                _outMat = null;
                try { TimeEndPeriod(1); } catch { }
            }
        }
    }
}