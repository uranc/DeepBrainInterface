using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    [Description("Ultimate Zero-Allocation CPU Fast-Path. Hardware-locked, red-lined, and shape-safeguarded.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorCPU : Transform<Mat, Tuple<Mat, double>>, IDisposable
    {
        /* ───── Windows Kernel P/Invoke Hooks ───── */
        [DllImport("kernel32.dll")]
        static extern IntPtr GetCurrentThread();

        [DllImport("kernel32.dll")]
        static extern IntPtr SetThreadAffinityMask(IntPtr hThread, IntPtr dwThreadAffinityMask);

        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);

        const uint _MCW_DN = 0x03000000;   // Denormal control mask
        const uint _DN_FLUSH = 0x01000000; // Flush to zero

        /* ───── User Properties ───── */
        [Category("1. Model")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        [FileNameFilter("ONNX Models|*.onnx|All Files|*.*")]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("2. Dimensions (Must Match Input)")]
        public int BatchSize { get; set; } = 1;
        [Category("2. Dimensions (Must Match Input)")]
        public int TimePoints { get; set; } = 44;
        [Category("2. Dimensions (Must Match Input)")]
        public int Channels { get; set; } = 8;

        [Category("3. Hardware & Red-Lining")]
        [Description("The logical CPU core to lock the ONNX thread to. (e.g. 2 means Core 2). Keeps Windows OS scheduler away.")]
        public int CpuCoreAffinity { get; set; } = 2;

        [Category("3. Hardware & Red-Lining")]
        [Description("Runs the math EVERY frame to keep the CPU awake, but only outputs a valid result every Nth frame.")]
        public int RunInterval { get; set; } = 3;

        /* ───── State & Buffers ───── */
        private readonly object _inferenceLock = new object();
        private bool _isThreadPinned = false;
        private int _batchCounter = 0;

        private InferenceSession _session;
        private RunOptions _runOpts;
        private Stopwatch _timer = new Stopwatch();

        private float[] _bufIn, _bufOut;
        private GCHandle _hIn, _hOut;
        private Mat _resultMat;

        private OrtValue[] _inputValues;
        private OrtValue[] _outputValues;
        private string[] _inputNames;
        private string[] _outputNames;

        private int _inputStride;
        private readonly List<Mat> _inputCache = new List<Mat>(2);

        private void InitializeSession()
        {
            if (_session != null) return;
            _inputStride = TimePoints * Channels;

            // 1. Hardware Subnormal Float Flush 
            try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { }

            // 2. Process Escalation
            try
            {
                // Fully qualify System.Diagnostics to avoid colliding with your Process() method
                using (var process = System.Diagnostics.Process.GetCurrentProcess())
                {
                    process.PriorityClass = ProcessPriorityClass.RealTime;
                }
                System.Threading.Thread.CurrentThread.Priority = System.Threading.ThreadPriority.Highest;
            }
            catch { }

            // 3. Ultra-Lean ORT Config
            var opts = new SessionOptions
            {
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableCpuMemArena = true,
                EnableMemoryPattern = true
            };

            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            opts.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");

            _session = new InferenceSession(ModelPath, opts);
            _runOpts = new RunOptions();
            opts.Dispose();

            _inputNames = new string[] { _session.InputMetadata.Keys.First() };
            _outputNames = new string[] { _session.OutputMetadata.Keys.First() };

            long[] inShape = new long[] { BatchSize, TimePoints, Channels };
            int inSize = BatchSize * TimePoints * Channels;

            var outMeta = _session.OutputMetadata.Values.First();
            long[] outShape = new long[outMeta.Dimensions.Length];
            int outSize = 1;
            for (int i = 0; i < outShape.Length; i++)
            {
                long d = outMeta.Dimensions[i];
                if (d <= 0) d = (i == 0) ? BatchSize : 1;
                outShape[i] = d;
                outSize *= (int)d;
            }

            // 4. Pin memory and bind directly to OrtValues 
            _bufIn = new float[inSize];
            _hIn = GCHandle.Alloc(_bufIn, GCHandleType.Pinned);

            _bufOut = new float[outSize];
            _hOut = GCHandle.Alloc(_bufOut, GCHandleType.Pinned);

            var memInfo = OrtMemoryInfo.DefaultInstance;
            _inputValues = new OrtValue[] { OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufIn), inShape) };
            _outputValues = new OrtValue[] { OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufOut), outShape) };

            int cols = outSize / BatchSize;
            _resultMat = new Mat(BatchSize, cols, Depth.F32, 1, _hOut.AddrOfPinnedObject());

            // 5. Deep Warmup: Run 100 times to violently stretch the Arena Allocator
            for (int i = 0; i < 100; i++)
            {
                _session.Run(_runOpts, _inputNames, _inputValues, _outputNames, _outputValues);
            }
        }

        private Tuple<Mat, double> RunInference(IList<Mat> mats)
        {
            lock (_inferenceLock)
            {
                if (_session == null) InitializeSession();

                // Lock Thread Affinity
                if (!_isThreadPinned)
                {
                    SetThreadAffinityMask(GetCurrentThread(), new IntPtr(1 << CpuCoreAffinity));
                    _isThreadPinned = true;
                }

                unsafe
                {
                    float* dstBase = (float*)_hIn.AddrOfPinnedObject().ToPointer();
                    int expectedElements = TimePoints * Channels;

                    for (int i = 0; i < mats.Count; i++)
                    {
                        int actualElements = mats[i].Rows * mats[i].Cols;
                        if (actualElements != expectedElements)
                        {
                            throw new InvalidOperationException(
                                $"FATAL SHAPE MISMATCH: ONNX expects {expectedElements} elements ({TimePoints}x{Channels}), " +
                                $"but received {actualElements} elements ({mats[i].Rows}x{mats[i].Cols}).");
                        }

                        float* src = (float*)mats[i].Data.ToPointer();
                        float* dst = dstBase + (i * _inputStride);

                        if (mats[i].Rows == TimePoints && mats[i].Cols == Channels)
                        {
                            Buffer.MemoryCopy(src, dst, _inputStride * sizeof(float), _inputStride * sizeof(float));
                        }
                        else if (mats[i].Rows == Channels && mats[i].Cols == TimePoints)
                        {
                            for (int c = 0; c < Channels; c++)
                                for (int t = 0; t < TimePoints; t++)
                                    dst[(t * Channels) + c] = src[(c * TimePoints) + t];
                        }
                    }
                }

                // THE GHOST RUN: Always execute the math
                _timer.Restart();
                _session.Run(_runOpts, _inputNames, _inputValues, _outputNames, _outputValues);
                _timer.Stop();

                _batchCounter++;

                if (_batchCounter >= RunInterval)
                {
                    _batchCounter = 0;
                    return Tuple.Create(_resultMat, _timer.Elapsed.TotalMilliseconds);
                }

                // Silent drop for Ghost Runs
                return null;
            }
        }

        // --- NATIVE BONSAI OVERRIDE ---
        public override IObservable<Tuple<Mat, double>> Process(IObservable<Mat> source)
        {
            return source.Select(m =>
            {
                lock (_inferenceLock)
                {
                    _inputCache.Clear();
                    _inputCache.Add(m);
                    return RunInference(_inputCache);
                }
            }).Where(res => res != null); // Instantly drop invalid Ghost Runs
        }

        public void Dispose()
        {
            if (_hIn.IsAllocated) _hIn.Free();
            if (_hOut.IsAllocated) _hOut.Free();

            if (_inputValues != null) foreach (var v in _inputValues) v?.Dispose();
            if (_outputValues != null) foreach (var v in _outputValues) v?.Dispose();

            _runOpts?.Dispose();
            _session?.Dispose();
        }
    }
}