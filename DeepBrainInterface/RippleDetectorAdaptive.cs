using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Adaptive Detector with Unmanaged Ring Buffer. ZERO ALLOCATIONS.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorAdaptive : IDisposable
    {
        [Category("Model")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        [FileNameFilter("ONNX Models|*.onnx|All Files|*.*")]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("Model Dimensions")] public int TimePoints { get; set; } = 44;
        [Category("Model Dimensions")] public int Channels { get; set; } = 8;

        [Category("Downsampling")]
        [Description("How many raw samples to skip between each point fed into the model.")]
        public int DownsampleFactor { get; set; } = 12;

        [Category("System")] public int RingBufferCapacity { get; set; } = 30000;
        [Category("System")] public bool HighPriorityThread { get; set; } = false;

        [Category("Logic")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public RippleStateMachineMatBool StateMachine { get; set; } = new RippleStateMachineMatBool();

        [Category("Logic")] public bool DetectionEnabled { get; set; } = true;
        [Category("Logic")] public int KBelowGate { get; set; } = 5;
        [Category("Logic")] public int KAtGate { get; set; } = 1;

        private readonly object _inferenceLock = new object();

        private InferenceSession _session;
        private OrtIoBinding _binding;
        private RunOptions _runOpts;

        private float[] _bufIn, _bufOut, _ringBuffer, _snapshotBuffer;
        private GCHandle _hIn, _hOut, _hRingBuffer, _hSnapshotBuffer;
        private Mat _snapshotMat;

        private int _headIndex = 0;
        private int _strideCounter = 0;
        private int _currentK = 1;

        private void Initialise()
        {
            if (_session != null) return;

            if (HighPriorityThread)
            {
                try { Thread.CurrentThread.Priority = ThreadPriority.Highest; }
                catch { }
            }

            var opts = new SessionOptions
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                EnableCpuMemArena = true
            };

            _session = new InferenceSession(ModelPath, opts);
            _binding = _session.CreateIoBinding();
            _runOpts = new RunOptions();

            int inputFloats = TimePoints * Channels;
            _bufIn = new float[inputFloats];
            _hIn = GCHandle.Alloc(_bufIn, GCHandleType.Pinned);

            _bufOut = new float[2];
            _hOut = GCHandle.Alloc(_bufOut, GCHandleType.Pinned);

            var memInfo = OrtMemoryInfo.DefaultInstance;

            // THE FIX: Using Memory<float> precisely as you originally designed it.
            using (var inOrt = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufIn), new long[] { 1, TimePoints, Channels }))
            using (var outOrt = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_bufOut), new long[] { 1, 2 }))
            {
                _binding.BindInput(_session.InputMetadata.Keys.First(), inOrt);
                _binding.BindOutput(_session.OutputMetadata.Keys.First(), outOrt);
            }

            _ringBuffer = new float[RingBufferCapacity * Channels];
            _hRingBuffer = GCHandle.Alloc(_ringBuffer, GCHandleType.Pinned);

            _snapshotBuffer = new float[Channels * TimePoints];
            _hSnapshotBuffer = GCHandle.Alloc(_snapshotBuffer, GCHandleType.Pinned);
            _snapshotMat = new Mat(Channels, TimePoints, Depth.F32, 1, _hSnapshotBuffer.AddrOfPinnedObject());

            _currentK = KBelowGate;

            _session.RunWithBinding(_runOpts, _binding);
        }

        public IObservable<RippleOut> Process(IObservable<Mat> source)
            => source.SelectMany(m => ExecuteSafe(m, true));

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
            => source.SelectMany(t => ExecuteSafe(t.Item1, t.Item2));

        private RippleOut[] ExecuteSafe(Mat rawData, bool bnoOk)
        {
            lock (_inferenceLock)
            {
                if (_session == null) Initialise();

                UpdateRingBufferUnsafe(rawData);

                if (_currentK > 1)
                {
                    _strideCounter++;
                    if (_strideCounter < _currentK)
                    {
                        return Array.Empty<RippleOut>();
                    }
                    _strideCounter = 0;
                }

                PrepareInputUnsafe();
                _session.RunWithBinding(_runOpts, _binding);

                float sigProb = _bufOut[0];
                float artProb = _bufOut[1];

                ExtractSnapshotUnsafe();

                StateMachine.DetectionEnabled = DetectionEnabled;
                bool gatesOpen = bnoOk && (artProb < StateMachine.GateThreshold);

                RippleOut result = StateMachine.Update(sigProb, artProb, gatesOpen, _snapshotMat);

                _currentK = (result.State == RippleState.Possible || result.State == RippleState.Ripple) ? KAtGate : KBelowGate;
                result.StrideUsed = _currentK;

                return new[] { result };
            }
        }

        private unsafe void UpdateRingBufferUnsafe(Mat mat)
        {
            float* srcPtr = (float*)mat.Data.ToPointer();
            float* ringPtr = (float*)_hRingBuffer.AddrOfPinnedObject().ToPointer();
            int cols = mat.Cols;

            for (int t = 0; t < cols; t++)
            {
                int ringIdx = (_headIndex + t) % RingBufferCapacity;
                int ringOffset = ringIdx * Channels;

                for (int c = 0; c < Channels; c++)
                {
                    ringPtr[ringOffset + c] = srcPtr[c * cols + t];
                }
            }
            _headIndex = (_headIndex + cols) % RingBufferCapacity;
        }

        private unsafe void PrepareInputUnsafe()
        {
            float* dstPtr = (float*)_hIn.AddrOfPinnedObject().ToPointer();
            float* ringPtr = (float*)_hRingBuffer.AddrOfPinnedObject().ToPointer();

            for (int t = 0; t < TimePoints; t++)
            {
                int stepsBack = (TimePoints - 1 - t) * DownsampleFactor;
                int ringIdx = _headIndex - 1 - stepsBack;

                while (ringIdx < 0) ringIdx += RingBufferCapacity;
                int ringOffset = ringIdx * Channels;

                for (int c = 0; c < Channels; c++)
                {
                    dstPtr[(t * Channels) + c] = ringPtr[ringOffset + c];
                }
            }
        }

        private unsafe void ExtractSnapshotUnsafe()
        {
            float* ringPtr = (float*)_hRingBuffer.AddrOfPinnedObject().ToPointer();
            float* dstPtr = (float*)_hSnapshotBuffer.AddrOfPinnedObject().ToPointer();

            for (int t = 0; t < TimePoints; t++)
            {
                int ringIdx = _headIndex - 1 - (TimePoints - 1 - t);
                while (ringIdx < 0) ringIdx += RingBufferCapacity;
                int ringOffset = ringIdx * Channels;

                for (int c = 0; c < Channels; c++)
                {
                    dstPtr[c * TimePoints + t] = ringPtr[ringOffset + c];
                }
            }
        }

        public void Dispose()
        {
            if (_hIn.IsAllocated) _hIn.Free();
            if (_hOut.IsAllocated) _hOut.Free();
            if (_hRingBuffer.IsAllocated) _hRingBuffer.Free();
            if (_hSnapshotBuffer.IsAllocated) _hSnapshotBuffer.Free();

            _binding?.Dispose();
            _runOpts?.Dispose();
            _session?.Dispose();
        }
    }
}