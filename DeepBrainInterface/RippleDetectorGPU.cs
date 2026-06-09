using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime;
using System.Runtime.InteropServices;
using System.Threading;

namespace DeepBrainInterface
{
    public struct InferenceResult
    {
        public Mat Data;
        public double LatencyMs;
    }

    [Combinator]
    [Description("Lean single-threaded CPU ONNX inference. Fused channel-major -> time-major transpose, " +
                 "pinned zero-copy IO binding, subnormal flush, optional core affinity, latency telemetry.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorGPU : IDisposable
    {
        // --- Subnormal (denormal) flush: avoids 10-100x slow paths on tiny floats ---
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        const uint _MCW_DN = 0x03000000;
        const uint _DN_FLUSH = 0x01000000;

        // --- System timer resolution + thread affinity ---
        [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod")] private static extern uint TimeBeginPeriod(uint uMilliseconds);
        [DllImport("winmm.dll", EntryPoint = "timeEndPeriod")] private static extern uint TimeEndPeriod(uint uMilliseconds);
        [DllImport("kernel32.dll")] static extern IntPtr GetCurrentThread();
        [DllImport("kernel32.dll")] static extern IntPtr SetThreadAffinityMask(IntPtr hThread, IntPtr dwThreadAffinityMask);

        [Category("1. Model Topology")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("1. Model Topology")] public int BatchSize { get; set; } = 1;
        [Category("1. Model Topology")] public int TimePoints { get; set; } = 44;
        [Category("1. Model Topology")] public int Channels { get; set; } = 8;

        [Category("2. Hardware Tuning")]
        [Description("Pin the inference thread to this CPU core. Set to -1 to disable affinity.")]
        public int TargetCore { get; set; } = 7;

        [Category("2. Hardware Tuning")]
        [Description("Let ORT busy-spin between ops instead of sleeping. Lower latency, higher CPU use.")]
        public bool AllowSpinning { get; set; } = true;

        [Category("2. Hardware Tuning")]
        [Description("Raise the inference thread to Highest priority within the process priority band.")]
        public bool HighThreadPriority { get; set; } = true;

        [Category("2. Hardware Tuning")]
        [Description("Process priority class. RealTime (base >=16) escapes the normal dynamic scheduling band " +
                     "and prevents the 100ms+ descheduling stalls. Safe here because TargetCore confines the hot " +
                     "loop to a single core, leaving the rest for the OS. Set to High if RealTime is too aggressive.")]
        public ProcessPriorityClass ProcessPriority { get; set; } = ProcessPriorityClass.RealTime;

        private readonly object _inferenceLock = new object();
        private InferenceSession _session;
        private OrtIoBinding _binding;
        private RunOptions _runOpts;
        private OrtValue _valIn, _valOut;
        private float[] _bufIn, _bufOut;
        private GCHandle _hIn, _hOut;
        private Mat _outMat;
        private bool _isInitialized;

        private int _inputStride;          // floats per sample = TimePoints * Channels
        private int _expectedInputBytes;   // bytes per sample (time-major fast-path copy)
        private int _outCols;
        private readonly Stopwatch _timer = new Stopwatch();

        private void Initialize()
        {
            // 1. Flush subnormals on THIS (the inference) thread.
            try { uint c = 0; _controlfp_s(ref c, _DN_FLUSH, _MCW_DN); } catch { }

            // 1b. Tell the GC to avoid expensive blocking Gen2 collections while acquiring.
            //     This attacks the *other* cause of long stalls (GC pauses) that priority alone
            //     cannot mask, since the GC suspends managed threads regardless of priority.
            try { GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency; } catch { }

            // 2. Tighten the OS scheduler tick.
            TimeBeginPeriod(1);

            // 3. Deterministic, single-threaded, fully-optimized CPU session.
            var opts = new SessionOptions
            {
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                EnableCpuMemArena = true,
                EnableMemoryPattern = true
            };
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            opts.AddSessionConfigEntry("session.execution_mode", "ORT_SEQUENTIAL");
            opts.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", AllowSpinning ? "1" : "0");

            _session = new InferenceSession(ModelPath, opts);
            _binding = _session.CreateIoBinding();
            _runOpts = new RunOptions();
            opts.Dispose();

            _inputStride = TimePoints * Channels;
            _expectedInputBytes = _inputStride * sizeof(float);

            // 4. Pinned, zero-copy input tensor [Batch, Time, Channels].
            var memInfo = OrtMemoryInfo.DefaultInstance;
            _bufIn = new float[BatchSize * _inputStride];
            _hIn = GCHandle.Alloc(_bufIn, GCHandleType.Pinned);
            _valIn = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufIn),
                new long[] { BatchSize, TimePoints, Channels });
            _binding.BindInput(_session.InputMetadata.Keys.First(), _valIn);

            // 5. Output shape from model metadata (robust to multi-element outputs).
            var outMeta = _session.OutputMetadata.Values.First();
            long[] outShape = new long[outMeta.Dimensions.Length];
            int outSize = 1;
            for (int i = 0; i < outShape.Length; i++)
            {
                long d = outMeta.Dimensions[i];
                if (d <= 0) d = (i == 0) ? BatchSize : 1;   // pin dynamic dims
                outShape[i] = d;
                outSize *= (int)d;
            }
            _outCols = outSize / BatchSize;

            _bufOut = new float[outSize];
            _hOut = GCHandle.Alloc(_bufOut, GCHandleType.Pinned);
            _valOut = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufOut), outShape);
            _binding.BindOutput(_session.OutputMetadata.Keys.First(), _valOut);

            // Mat header over the pinned output buffer; cloned per emission for downstream safety.
            _outMat = new Mat(BatchSize, _outCols, Depth.F32, 1, _hOut.AddrOfPinnedObject());

            // 6. Deep warmup: JIT kernels + populate the arena before real-time data flows.
            for (int i = 0; i < 50; i++) _session.RunWithBinding(_runOpts, _binding);

            // 7. Real-time scheduling. The process priority must reach the real-time band (>=16) to
            //    avoid the 100ms+ descheduling stalls seen under Normal/High priority. Doing it here
            //    means you no longer need to set it by hand in Task Manager. The single-core affinity
            //    below keeps the other cores free so the OS stays responsive under RealTime.
            try
            {
                using (var proc = Process.GetCurrentProcess())
                    proc.PriorityClass = ProcessPriority;
            }
            catch { }

            if (TargetCore >= 0 && TargetCore < 64)
                SetThreadAffinityMask(GetCurrentThread(), (IntPtr)(1L << TargetCore));
            if (HighThreadPriority)
                try { Thread.CurrentThread.Priority = ThreadPriority.Highest; } catch { }

            _isInitialized = true;
        }

        public IObservable<InferenceResult> Process(IObservable<Mat> source)
        {
            return source.Select(m =>
            {
                lock (_inferenceLock)
                {
                    if (!_isInitialized) Initialize();

                    // Fused copy + transpose into the pinned input tensor.
                    // The model expects TIME-MAJOR [Batch, TimePoints, Channels]: the 8 channel
                    // values for a timepoint are contiguous (dst[t*Channels + c]).
                    // Rhd2164 amplifier windows are CHANNEL-MAJOR [Channels x TimePoints]
                    // (rows = channels), so transpose on the fly; no extra allocation.
                    unsafe
                    {
                        float* dst = (float*)_hIn.AddrOfPinnedObject().ToPointer();
                        byte* src = (byte*)m.Data.ToPointer();
                        int step = m.Step;

                        if (m.Rows == Channels && m.Cols == TimePoints)
                        {
                            // [Channels x TimePoints] channel-major -> time-major
                            for (int c = 0; c < Channels; c++)
                            {
                                float* row = (float*)(src + c * step);
                                for (int t = 0; t < TimePoints; t++)
                                    dst[t * Channels + c] = row[t];
                            }
                        }
                        else if (m.Rows == TimePoints && m.Cols == Channels)
                        {
                            // [TimePoints x Channels] already time-major
                            int rowBytes = Channels * sizeof(float);
                            if (step == rowBytes)
                                Buffer.MemoryCopy(src, dst, _expectedInputBytes, _expectedInputBytes);
                            else
                                for (int t = 0; t < TimePoints; t++)
                                    Buffer.MemoryCopy(src + t * step, dst + t * Channels, rowBytes, rowBytes);
                        }
                        else
                        {
                            throw new InvalidOperationException(
                                $"Unexpected input shape {m.Rows}x{m.Cols}; expected " +
                                $"{Channels}x{TimePoints} (channel-major) or {TimePoints}x{Channels} (time-major).");
                        }
                    }

                    _timer.Restart();
                    _session.RunWithBinding(_runOpts, _binding);
                    _timer.Stop();

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
                _outMat?.Dispose();
                _valIn?.Dispose();
                _valOut?.Dispose();
                _binding?.Dispose();
                _runOpts?.Dispose();
                _session?.Dispose();

                if (_hIn.IsAllocated) _hIn.Free();
                if (_hOut.IsAllocated) _hOut.Free();

                _outMat = null;
                _valIn = null; _valOut = null;
                _binding = null; _runOpts = null; _session = null;
                _isInitialized = false;

                try { TimeEndPeriod(1); } catch { }
            }
        }
    }
}
