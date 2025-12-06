using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms.Design;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Ripple Detector Switch: Runs Gate (Small) or Active (Large) model based on State Logic.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorSwitch
    {
        // ==============================================================================
        // 1. MODEL A: GATE (Small/Fast)
        // ==============================================================================
        [Category("Model A: Gate (Relaxed)")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        [Description("Small model used when scanning (NoRipple).")]
        public string ModelPathGate { get; set; } = @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector_small.onnx";

        [Category("Model A: Gate (Relaxed)")]
        public int GateInputSize { get; set; } = 44;

        [Category("Model A: Gate (Relaxed)")]
        [Description("Downsample step for Gate model (e.g. 12 for ~2.5kHz).")]
        public int GateDownsample { get; set; } = 12;

        // ==============================================================================
        // 2. MODEL B: ACTIVE (Large/Accurate)
        // ==============================================================================
        [Category("Model B: Active (Focus)")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        [Description("Large model used when verifying (Possible/Ripple).")]
        public string ModelPathActive { get; set; } = @"C:\Users\angel\Documents\BonsaiFiles\frozen_models\ripple_detector_large.onnx";

        [Category("Model B: Active (Focus)")]
        public int ActiveInputSize { get; set; } = 560;

        [Category("Model B: Active (Focus)")]
        [Description("Downsample step for Active model (e.g. 1 for 30kHz).")]
        public int ActiveDownsample { get; set; } = 1;

        // ==============================================================================
        // 3. SHARED CONFIG
        // ==============================================================================
        [Category("Shared Config")]
        public int BatchSize { get; set; } = 1;

        [Category("Shared Config")]
        public int Channels { get; set; } = 8;

        [Category("Shared Config")]
        [Description("Internal Ring Buffer size (samples). Must be > Max Window needed.")]
        public int RingBufferCapacity { get; set; } = 30000;

        // ==============================================================================
        // 4. ACTIVE SENSING & LOGIC
        // ==============================================================================
        [Category("Logic")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public RippleStateMachineMatBool StateMachine { get; set; } = new RippleStateMachineMatBool();

        [Category("Active Sensing")]
        public bool DetectionEnabled { get; set; } = true;

        [Category("Active Sensing")]
        [Description("Stride K for Gate Model (Relaxed).")]
        public int KBelowGate { get; set; } = 5;

        [Category("Active Sensing")]
        [Description("Stride K for Active Model (Focus).")]
        public int KAtGate { get; set; } = 1;


        // ==============================================================================
        // INTERNAL STATE
        // ==============================================================================

        // Sessions
        InferenceSession _sessionGate;
        InferenceSession _sessionActive;
        OrtIoBinding _bindGate;
        OrtIoBinding _bindActive;
        RunOptions _runOptions;

        // Pinned IO Buffers
        OrtValue _valGateIn, _valGateOut;
        float[] _bufGateIn, _bufGateOut;

        OrtValue _valActiveIn, _valActiveOut;
        float[] _bufActiveIn, _bufActiveOut;

        // Ring Buffer (One per batch index)
        List<float[]> _ringBuffers;
        int _headIndex;
        int _strideCounter;
        int _currentK = 1;

        // State Tracker
        bool _useActiveModel = false;

        struct InfResult { public float Sig; public float Art; }
        struct InputPackage { public Mat[] Mats; public bool BnoOk; }

        private void Initialise()
        {
            if (_sessionGate != null) return;

            if (!File.Exists(ModelPathGate)) throw new FileNotFoundException("Gate Model not found", ModelPathGate);
            if (!File.Exists(ModelPathActive)) throw new FileNotFoundException("Active Model not found", ModelPathActive);

            // 1. Init Ring Buffers
            _ringBuffers = new List<float[]>();
            for (int i = 0; i < BatchSize; i++)
            {
                _ringBuffers.Add(new float[RingBufferCapacity * Channels]);
            }
            _headIndex = 0;

            // 2. CPU Options (Single Threaded)
            var opts = new SessionOptions
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                EnableCpuMemArena = true
            };

            // 3. Load Sessions
            _sessionGate = new InferenceSession(ModelPathGate, opts);
            _sessionActive = new InferenceSession(ModelPathActive, opts);
            _runOptions = new RunOptions();

            var memInfo = OrtMemoryInfo.DefaultInstance;

            // 4. Setup GATE Binding
            _bufGateIn = new float[BatchSize * GateInputSize * Channels];
            _valGateIn = OrtValue.CreateTensorValueFromMemory<float>(memInfo, _bufGateIn, new long[] { BatchSize, GateInputSize, Channels });
            _bufGateOut = AllocateOutputBuffer(_sessionGate, GateInputSize, out long[] shapeGate);
            _valGateOut = OrtValue.CreateTensorValueFromMemory<float>(memInfo, _bufGateOut, shapeGate);

            _bindGate = _sessionGate.CreateIoBinding();
            _bindGate.BindInput(_sessionGate.InputMetadata.Keys.First(), _valGateIn);
            _bindGate.BindOutput(_sessionGate.OutputMetadata.Keys.First(), _valGateOut);

            // 5. Setup ACTIVE Binding
            _bufActiveIn = new float[BatchSize * ActiveInputSize * Channels];
            _valActiveIn = OrtValue.CreateTensorValueFromMemory<float>(memInfo, _bufActiveIn, new long[] { BatchSize, ActiveInputSize, Channels });

            _bufActiveOut = AllocateOutputBuffer(_sessionActive, ActiveInputSize, out long[] shapeActive);
            _valActiveOut = OrtValue.CreateTensorValueFromMemory<float>(memInfo, _bufActiveOut, shapeActive);

            _bindActive = _sessionActive.CreateIoBinding();
            _bindActive.BindInput(_sessionActive.InputMetadata.Keys.First(), _valActiveIn);
            _bindActive.BindOutput(_sessionActive.OutputMetadata.Keys.First(), _valActiveOut);

            // 6. Warmup Both
            _sessionGate.RunWithBinding(_runOptions, _bindGate);
            _sessionActive.RunWithBinding(_runOptions, _bindActive);

            _currentK = KBelowGate;
        }

        private float[] AllocateOutputBuffer(InferenceSession session, int timePts, out long[] shape)
        {
            var name = session.OutputMetadata.Keys.First();
            var dims = session.OutputMetadata[name].Dimensions;
            shape = new long[dims.Length];
            long size = 1;

            for (int i = 0; i < dims.Length; i++)
            {
                long d = dims[i];
                if (d <= 0)
                {
                    if (i == 0) d = BatchSize;
                    else if (dims.Length == 3 && i == 1) d = timePts;
                    else d = 1;
                }
                // Rank 3 Fix
                if (dims.Length == 3 && i == 1 && d == 1 && timePts > 1) d = timePts;
                shape[i] = d;
                size *= d;
            }
            return new float[size];
        }

        // ==============================================================================
        // OVERLOADS
        // ==============================================================================
        public IObservable<RippleOut> Process(IObservable<Mat> source) => ProcessInternal(source.Select(m => new InputPackage { Mats = new[] { m }, BnoOk = true }));
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source) => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1 }, BnoOk = t.Item2 }));
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, Mat>> source) => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1, t.Item2 }, BnoOk = true }));
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, Mat, Mat>> source) => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1, t.Item2, t.Item3 }, BnoOk = true }));
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, Mat, Mat, Mat>> source) => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1, t.Item2, t.Item3, t.Item4 }, BnoOk = true }));
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source) => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1.Item1, t.Item1.Item2 }, BnoOk = t.Item2 }));
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat, Mat>, bool>> source) => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1.Item1, t.Item1.Item2, t.Item1.Item3 }, BnoOk = t.Item2 }));
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat, Mat, Mat>, bool>> source) => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1.Item1, t.Item1.Item2, t.Item1.Item3, t.Item1.Item4 }, BnoOk = t.Item2 }));

        // ==============================================================================
        // MAIN PROCESSING LOOP
        // ==============================================================================
        private IObservable<RippleOut> ProcessInternal(IObservable<InputPackage> source)
        {
            return source.Select(input =>
            {
                Initialise();

                // 1. WRITE TO RING BUFFER (Always)
                UpdateRingBuffer(input.Mats);

                // 2. STRIDE GATE
                int k = _currentK;
                bool runInference = (k <= 1);
                if (k > 1)
                {
                    _strideCounter++;
                    if (_strideCounter >= k) { _strideCounter = 0; runInference = true; }
                }

                if (!runInference) return null;

                // 3. DECIDE MODEL
                // Relaxed (K=5) -> Gate Model. Focused (K=1) -> Active Model.
                // This uses the state from the PREVIOUS frame.
                bool useActive = (_currentK == KAtGate);
                _useActiveModel = useActive;

                // 4. RUN INFERENCE
                InfResult res;
                if (useActive)
                {
                    PrepareInferenceInput(_bufActiveIn, ActiveInputSize, ActiveDownsample);
                    _sessionActive.RunWithBinding(_runOptions, _bindActive);
                    res = ReadOutput(_bufActiveOut);
                }
                else
                {
                    PrepareInferenceInput(_bufGateIn, GateInputSize, GateDownsample);
                    _sessionGate.RunWithBinding(_runOptions, _bindGate);
                    res = ReadOutput(_bufGateOut);
                }

                // 5. LOGIC
                StateMachine.DetectionEnabled = DetectionEnabled;

                // Snapshot logic: Get High-Res view for trigger
                Mat snapshot = ExtractHighResSnapshot();

                RippleOut output = StateMachine.Update(res.Sig, res.Art, input.BnoOk, snapshot);

                // 6. FEEDBACK
                bool artifactOk = res.Art < StateMachine.ArtifactThreshold;
                bool gatesOpen = input.BnoOk && artifactOk;

                if (!gatesOpen)
                {
                    _currentK = KBelowGate; // Relax
                }
                else if (output.State == RippleState.Possible)
                {
                    _currentK = KAtGate; // Focus
                }
                else
                {
                    _currentK = KBelowGate; // Relax
                }

                output.StrideUsed = _currentK;
                return output;
            })
            .Where(o => o != null);
        }

        // --------------------------------------------------------------------------
        // RING BUFFER UTILS
        // --------------------------------------------------------------------------
        private void UpdateRingBuffer(Mat[] mats)
        {
            int count = Math.Min(BatchSize, mats.Length);
            int cols = mats[0].Cols;
            unsafe
            {
                for (int b = 0; b < count; b++)
                {
                    float* src = (float*)mats[b].Data.ToPointer();
                    float[] ring = _ringBuffers[b];

                    for (int t = 0; t < cols; t++)
                    {
                        int ringIdx = (_headIndex + t) % RingBufferCapacity;
                        for (int c = 0; c < Channels; c++)
                        {
                            // Planar Input [C, T] -> Interleaved Ring [T, C]
                            ring[ringIdx * Channels + c] = src[c * cols + t];
                        }
                    }
                }
            }
            _headIndex = (_headIndex + cols) % RingBufferCapacity;
        }

        private void PrepareInferenceInput(float[] dstBuffer, int inputSize, int step)
        {
            unsafe
            {
                fixed (float* dstPtr = dstBuffer)
                {
                    for (int b = 0; b < BatchSize; b++)
                    {
                        if (b >= _ringBuffers.Count) break;
                        float[] ring = _ringBuffers[b];
                        int batchOffset = b * inputSize * Channels;

                        for (int t = 0; t < inputSize; t++)
                        {
                            int stepsBack = (inputSize - 1 - t) * step;
                            int ringIdx = _headIndex - 1 - stepsBack;
                            while (ringIdx < 0) ringIdx += RingBufferCapacity;

                            for (int c = 0; c < Channels; c++)
                            {
                                // N-C-W layout for Model
                                dstPtr[batchOffset + (c * inputSize) + t] = ring[ringIdx * Channels + c];
                            }
                        }
                    }
                }
            }
        }

        private InfResult ReadOutput(float[] buffer)
        {
            int stride = buffer.Length / BatchSize;
            float sig = buffer[stride - 1];
            float art = 0;
            if (BatchSize > 1) art = buffer[(stride * 2) - 1];

            return new InfResult { Sig = sig, Art = art };
        }

        private Mat ExtractHighResSnapshot()
        {
            // Extracts ActiveInputSize (High Res) from Batch 0
            var m = new Mat(Channels, ActiveInputSize, Depth.F32, 1);
            float[] ring = _ringBuffers[0];
            unsafe
            {
                float* dst = (float*)m.Data.ToPointer();
                for (int t = 0; t < ActiveInputSize; t++)
                {
                    int ringIdx = _headIndex - 1 - (ActiveInputSize - 1 - t);
                    while (ringIdx < 0) ringIdx += RingBufferCapacity;
                    for (int c = 0; c < Channels; c++)
                        dst[c * ActiveInputSize + t] = ring[ringIdx * Channels + c];
                }
            }
            return m;
        }

        ~RippleDetectorSwitch()
        {
            _valGateIn?.Dispose(); _valGateOut?.Dispose();
            _valActiveIn?.Dispose(); _valActiveOut?.Dispose();
            _bindGate?.Dispose(); _bindActive?.Dispose();
            _runOptions?.Dispose(); _sessionGate?.Dispose(); _sessionActive?.Dispose();
        }
    }
}