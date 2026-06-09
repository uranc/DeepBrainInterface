using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace DeepBrainInterface
{
    public struct SuperNodeResult
    {
        public float Probability;
        public double LatencyMs;
        public int InferencesSkipped;
    }

    [Combinator]
    [Description("Decoupled inference: acquisition thread writes a lock-free ring; a pinned, RealTime " +
                 "background thread spin-polls the freshest window and runs zero-alloc ONNX inference.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorSuperNode : IDisposable
    {
        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern uint _controlfp_s(ref uint currentControl, uint newControl, uint mask);
        const uint _MCW_DN = 0x03000000;
        const uint _DN_FLUSH = 0x01000000;

        [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod")] private static extern uint TimeBeginPeriod(uint ms);
        [DllImport("winmm.dll", EntryPoint = "timeEndPeriod")] private static extern uint TimeEndPeriod(uint ms);
        [DllImport("kernel32.dll")] static extern IntPtr GetCurrentThread();
        [DllImport("kernel32.dll")] static extern IntPtr SetThreadAffinityMask(IntPtr hThread, IntPtr dwThreadAffinityMask);

        [Category("1. Model")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("1. Model")] public int TimePoints { get; set; } = 44;
        [Category("1. Model")] public int Channels { get; set; } = 8;

        [Category("2. Hardware Tuning")]
        [Description("Run inference whenever this many new samples have arrived (stride). 3 ~= 833 Hz at 2.5 kHz.")]
        public int InferenceStride { get; set; } = 3;

        [Category("2. Hardware Tuning")]
        [Description("Pin the inference thread to this core. -1 disables affinity. Busy-spin pegs this core.")]
        public int TargetCore { get; set; } = 7;

        [Category("2. Hardware Tuning")]
        [Description("Busy-spin for lowest pickup latency (pegs TargetCore). False = SpinOnce (yields, lower CPU).")]
        public bool BusySpin { get; set; } = true;

        [Category("2. Hardware Tuning")]
        [Description("Process priority. RealTime keeps the acquisition + inference threads off the dynamic band.")]
        public ProcessPriorityClass ProcessPriority { get; set; } = ProcessPriorityClass.RealTime;

        // Ring must be a power of two and much larger than TimePoints.
        private const int RingSize = 1024;
        private float[] _ring;
        private GCHandle _hRing;
        private volatile int _writeHead;
        private long _written;                 // total samples written (warmup gate)
        private int _channels, _timePoints;
        private CancellationTokenSource _cts;

        public IObservable<SuperNodeResult> Process(IObservable<Mat> source)
        {
            return Observable.Create<SuperNodeResult>(observer =>
            {
                _channels = Channels;
                _timePoints = TimePoints;
                _ring = new float[RingSize * _channels];
                _hRing = GCHandle.Alloc(_ring, GCHandleType.Pinned);
                _writeHead = 0;
                _written = 0;
                _cts = new CancellationTokenSource();

                // Writer: cheap, lock-free, runs on the acquisition thread. Channel-major [Channels x N] in.
                var sub = source.Subscribe(mat =>
                {
                    int samples = mat.Cols;
                    int step = mat.Step;
                    unsafe
                    {
                        byte* baseB = (byte*)mat.Data.ToPointer();
                        float* ring = (float*)_hRing.AddrOfPinnedObject().ToPointer();
                        for (int t = 0; t < samples; t++)
                        {
                            int head = _writeHead;
                            int off = head * _channels;
                            for (int c = 0; c < _channels; c++)
                                ring[off + c] = *(float*)(baseB + c * step + t * sizeof(float));
                            _writeHead = (head + 1) & (RingSize - 1);
                            _written++;
                        }
                    }
                }, observer.OnError);

                var worker = new Thread(() => RunPollingInference(observer, _cts.Token))
                {
                    IsBackground = true,
                    Priority = ThreadPriority.Highest,
                    Name = "RippleSuperNodeInference"
                };
                worker.Start();

                return Disposable.Create(() =>
                {
                    _cts.Cancel();
                    worker.Join(500);
                    sub.Dispose();
                    if (_hRing.IsAllocated) _hRing.Free();
                });
            });
        }

        private unsafe void RunPollingInference(IObserver<SuperNodeResult> observer, CancellationToken token)
        {
            try { uint cc = 0; _controlfp_s(ref cc, _DN_FLUSH, _MCW_DN); } catch { }
            TimeBeginPeriod(1);
            try
            {
                using (var proc = System.Diagnostics.Process.GetCurrentProcess())
                    proc.PriorityClass = ProcessPriority;
            }
            catch { }
            if (TargetCore >= 0 && TargetCore < 64)
                SetThreadAffinityMask(GetCurrentThread(), (IntPtr)(1L << TargetCore));

            int stride = _timePoints * _channels;
            int strideSamples = InferenceStride < 1 ? 1 : InferenceStride;

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
            opts.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");

            var session = new InferenceSession(ModelPath, opts);
            var binding = session.CreateIoBinding();
            var runOpts = new RunOptions();
            opts.Dispose();

            // Pinned, zero-copy IO bound once.
            float[] inBuf = new float[stride];
            var hIn = GCHandle.Alloc(inBuf, GCHandleType.Pinned);
            float[] outBuf = new float[1];
            var hOut = GCHandle.Alloc(outBuf, GCHandleType.Pinned);
            var mem = OrtMemoryInfo.DefaultInstance;
            var valIn = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(inBuf), new long[] { 1, _timePoints, _channels });
            var valOut = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(outBuf), new long[] { 1, 1 });
            binding.BindInput(session.InputMetadata.Keys.First(), valIn);
            binding.BindOutput(session.OutputMetadata.Keys.First(), valOut);

            for (int i = 0; i < 50; i++) session.RunWithBinding(runOpts, binding);

            float* ring = (float*)_hRing.AddrOfPinnedObject().ToPointer();
            float* dst = (float*)hIn.AddrOfPinnedObject().ToPointer();
            var sw = new Stopwatch();
            var spinner = new SpinWait();
            int lastInferHead = 0;
            int skipped = 0;

            try
            {
                while (!token.IsCancellationRequested)
                {
                    // Warmup: need at least one full window written.
                    if (Volatile.Read(ref _written) < _timePoints)
                    {
                        if (BusySpin) Thread.SpinWait(64); else spinner.SpinOnce();
                        continue;
                    }

                    int head = _writeHead;
                    int newSamples = (head - lastInferHead) & (RingSize - 1);
                    if (newSamples < strideSamples)
                    {
                        if (BusySpin) Thread.SpinWait(64); else spinner.SpinOnce();
                        continue;
                    }
                    if (newSamples > strideSamples * 2) skipped += (newSamples / strideSamples) - 1;

                    // Copy the freshest window [head - TimePoints, head), wrap-safe.
                    int start = (head - _timePoints) & (RingSize - 1);
                    int firstSamples = Math.Min(_timePoints, RingSize - start);
                    Buffer.MemoryCopy(ring + start * _channels, dst,
                        stride * sizeof(float), firstSamples * _channels * sizeof(float));
                    if (firstSamples < _timePoints)
                    {
                        int remFloats = (_timePoints - firstSamples) * _channels;
                        Buffer.MemoryCopy(ring, dst + firstSamples * _channels,
                            remFloats * sizeof(float), remFloats * sizeof(float));
                    }

                    sw.Restart();
                    session.RunWithBinding(runOpts, binding);
                    sw.Stop();

                    observer.OnNext(new SuperNodeResult
                    {
                        Probability = outBuf[0],
                        LatencyMs = sw.Elapsed.TotalMilliseconds,
                        InferencesSkipped = skipped
                    });
                    lastInferHead = head;
                }
            }
            finally
            {
                valIn.Dispose();
                valOut.Dispose();
                binding.Dispose();
                runOpts.Dispose();
                session.Dispose();
                if (hIn.IsAllocated) hIn.Free();
                if (hOut.IsAllocated) hOut.Free();
                TimeEndPeriod(1);
            }
        }

        public void Dispose()
        {
            _cts?.Cancel();
        }
    }
}
