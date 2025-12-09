//using Bonsai;
//using Microsoft.ML.OnnxRuntime;
//using Microsoft.ML.OnnxRuntime.Tensors;
//using OpenCV.Net;
//using System;
//using System.Collections.Generic;
//using System.ComponentModel;
//using System.Diagnostics;
//using System.Drawing.Design;
//using System.IO;
//using System.Linq;
//using System.Reactive.Linq;
//using System.Runtime.InteropServices;
//using System.Threading;
//using System.Windows.Forms.Design;

//namespace DeepBrainInterface
//{
//    [Combinator]
//    [Description("Ripple Detector Switch: Optimized for Real-Time (Pinned Memory, Zero-Copy ORT).")]
//    [WorkflowElementCategory(ElementCategory.Transform)]
//    public class RippleDetectorSwitch
//    {
//        // ==============================================================================
//        // 1. MODEL CONFIGURATION
//        // ==============================================================================
//        [Category("Model A: Gate")]
//        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
//        public string ModelPathGate { get; set; } = @"C:\RippleModels\gate_small.onnx";
//        [Category("Model A: Gate")] public int GateInputSize { get; set; } = 44;
//        [Category("Model A: Gate")] public int GateDownsample { get; set; } = 12;

//        [Category("Model B: Active")]
//        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
//        public string ModelPathActive { get; set; } = @"C:\RippleModels\active_large.onnx";
//        [Category("Model B: Active")] public int ActiveInputSize { get; set; } = 560;
//        [Category("Model B: Active")] public int ActiveDownsample { get; set; } = 1;

//        // ==============================================================================
//        // 2. SYSTEM CONFIGURATION
//        // ==============================================================================
//        [Category("System")] public int BatchSize { get; set; } = 1;
//        [Category("System")] public int Channels { get; set; } = 8;
//        [Category("System")] public int RingBufferCapacity { get; set; } = 30000;

//        [Category("System")]
//        [Description("If true, attempts to set the processing thread to Highest Priority.")]
//        public bool HighPriorityThread { get; set; } = false;

//        // ==============================================================================
//        // 3. LOGIC
//        // ==============================================================================
//        [Category("Logic")]
//        [TypeConverter(typeof(ExpandableObjectConverter))]
//        public RippleStateMachineMatBool StateMachine { get; set; } = new RippleStateMachineMatBool();

//        [Category("Logic")] public bool DetectionEnabled { get; set; } = true;
//        [Category("Logic")] public int KBelowGate { get; set; } = 5;
//        [Category("Logic")] public int KAtGate { get; set; } = 1;

//        // ==============================================================================
//        // INTERNAL STATE (PINNED)
//        // ==============================================================================

//        // ONNX Resources
//        InferenceSession _sessionGate, _sessionActive;
//        OrtIoBinding _bindGate, _bindActive;
//        RunOptions _runOptions;

//        // Pinned Buffers (GC Handles prevents memory from moving)
//        GCHandle _hGateIn, _hGateOut, _hActiveIn, _hActiveOut;
//        float[] _bufGateIn, _bufGateOut, _bufActiveIn, _bufActiveOut;
//        OrtValue _valGateIn, _valGateOut, _valActiveIn, _valActiveOut;

//        // Ring Buffers (List of arrays, also pinned)
//        List<float[]> _ringBuffers;
//        List<GCHandle> _hRingBuffers;

//        // Snapshot Buffer (Pre-allocated for output Mat)
//        float[] _snapshotBuffer;
//        GCHandle _hSnapshotBuffer;

//        int _headIndex;
//        int _strideCounter;
//        int _currentK = 1;

//        struct InfResult { public float Sig; public float Art; }
//        struct InputPackage { public Mat[] Mats; public bool BnoOk; }

//        private void Initialise()
//        {
//            if (_sessionGate != null) return;

//            // 1. Thread Priority Upgrade
//            if (HighPriorityThread)
//            {
//                try { Thread.CurrentThread.Priority = ThreadPriority.Highest; }
//                catch { /* Ignore if security exception */ }
//            }

//            // 2. Load Options
//            var opts = new SessionOptions
//            {
//                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
//                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
//                IntraOpNumThreads = 1, // Crucial for low latency single-batch
//                InterOpNumThreads = 1,
//                EnableCpuMemArena = true
//            };

//            _sessionGate = new InferenceSession(ModelPathGate, opts);
//            _sessionActive = new InferenceSession(ModelPathActive, opts);
//            _runOptions = new RunOptions();

//            // 3. Allocate & PIN Memory
//            // We pin everything so we can pass IntPtrs to ORT without copying.

//            // --- Gate ---
//            _bufGateIn = new float[BatchSize * GateInputSize * Channels];
//            _hGateIn = GCHandle.Alloc(_bufGateIn, GCHandleType.Pinned);

//            // Allocate Output (We need to calculate size first)
//            long[] shapeGate;
//            int gateOutLen = CalculateOutputSize(_sessionGate, GateInputSize, out shapeGate);
//            _bufGateOut = new float[gateOutLen];
//            _hGateOut = GCHandle.Alloc(_bufGateOut, GCHandleType.Pinned);

//            // Create OrtValues from Pinned Pointers
//            var memInfo = OrtMemoryInfo.DefaultInstance;
//            _valGateIn = OrtValue.CreateTensorValueFromMemory(memInfo, _hGateIn.AddrOfPinnedObject(),
//                         _bufGateIn.Length * sizeof(float), new long[] { BatchSize, GateInputSize, Channels });

//            _valGateOut = OrtValue.CreateTensorValueFromMemory(memInfo, _hGateOut.AddrOfPinnedObject(),
//                          _bufGateOut.Length * sizeof(float), shapeGate);

//            // Bind Gate
//            _bindGate = _sessionGate.CreateIoBinding();
//            _bindGate.BindInput(_sessionGate.InputMetadata.Keys.First(), _valGateIn);
//            _bindGate.BindOutput(_sessionGate.OutputMetadata.Keys.First(), _valGateOut);

//            // --- Active ---
//            _bufActiveIn = new float[BatchSize * ActiveInputSize * Channels];
//            _hActiveIn = GCHandle.Alloc(_bufActiveIn, GCHandleType.Pinned);

//            long[] shapeActive;
//            int activeOutLen = CalculateOutputSize(_sessionActive, ActiveInputSize, out shapeActive);
//            _bufActiveOut = new float[activeOutLen];
//            _hActiveOut = GCHandle.Alloc(_bufActiveOut, GCHandleType.Pinned);

//            _valActiveIn = OrtValue.CreateTensorValueFromMemory(memInfo, _hActiveIn.AddrOfPinnedObject(),
//                           _bufActiveIn.Length * sizeof(float), new long[] { BatchSize, ActiveInputSize, Channels });

//            _valActiveOut = OrtValue.CreateTensorValueFromMemory(memInfo, _hActiveOut.AddrOfPinnedObject(),
//                            _bufActiveOut.Length * sizeof(float), shapeActive);

//            _bindActive = _sessionActive.CreateIoBinding();
//            _bindActive.BindInput(_sessionActive.InputMetadata.Keys.First(), _valActiveIn);
//            _bindActive.BindOutput(_sessionActive.OutputMetadata.Keys.First(), _valActiveOut);

//            // --- Ring Buffers ---
//            _ringBuffers = new List<float[]>();
//            _hRingBuffers = new List<GCHandle>();
//            for (int i = 0; i < BatchSize; i++)
//            {
//                var arr = new float[RingBufferCapacity * Channels];
//                _ringBuffers.Add(arr);
//                _hRingBuffers.Add(GCHandle.Alloc(arr, GCHandleType.Pinned));
//            }

//            // --- Snapshot Buffer ---
//            _snapshotBuffer = new float[Channels * ActiveInputSize];
//            _hSnapshotBuffer = GCHandle.Alloc(_snapshotBuffer, GCHandleType.Pinned);

//            // Warmup
//            _sessionGate.RunWithBinding(_runOptions, _bindGate);
//            _sessionActive.RunWithBinding(_runOptions, _bindActive);
//            _currentK = KBelowGate;
//        }

//        private int CalculateOutputSize(InferenceSession session, int timePts, out long[] shape)
//        {
//            var name = session.OutputMetadata.Keys.First();
//            var dims = session.OutputMetadata[name].Dimensions;
//            shape = new long[dims.Length];
//            long size = 1;
//            for (int i = 0; i < dims.Length; i++)
//            {
//                long d = dims[i];
//                if (d <= 0)
//                {
//                    if (i == 0) d = BatchSize;
//                    else if (dims.Length == 3 && i == 1) d = timePts;
//                    else d = 1;
//                }
//                if (dims.Length == 3 && i == 1 && d == 1 && timePts > 1) d = timePts;
//                shape[i] = d;
//                size *= d;
//            }
//            return (int)size;
//        }

//        // ==============================================================================
//        // OVERLOADS (Boilerplate)
//        // ==============================================================================
//        public IObservable<RippleOut> Process(IObservable<Mat> source) => ProcessInternal(source.Select(m => new InputPackage { Mats = new[] { m }, BnoOk = true }));
//        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source) => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1 }, BnoOk = t.Item2 }));
//        // ... (Other overloads omitted for brevity, same pattern as before)

//        // ==============================================================================
//        // FAST PATH PROCESSING
//        // ==============================================================================
//        private IObservable<RippleOut> ProcessInternal(IObservable<InputPackage> source)
//        {
//            return source.Select(input =>
//            {
//                if (_sessionGate == null) Initialise();

//                // 1. Unsafe Ring Buffer Update (No Bounds Checks)
//                UpdateRingBufferUnsafe(input.Mats);

//                // 2. Stride Logic
//                int k = _currentK;
//                bool runInference = (k <= 1);
//                if (k > 1)
//                {
//                    _strideCounter++;
//                    if (_strideCounter >= k) { _strideCounter = 0; runInference = true; }
//                }

//                if (!runInference) return (RippleOut?)null;

//                // 3. Inference
//                bool useActive = (_currentK == KAtGate);
//                InfResult res;

//                if (useActive)
//                {
//                    // Copy Ring -> Pinned Input Buffer
//                    PrepareInputUnsafe(_bufActiveIn, ActiveInputSize, ActiveDownsample);
//                    // Zero-Copy Run
//                    _sessionActive.RunWithBinding(_runOptions, _bindActive);
//                    res = ReadOutput(_bufActiveOut);
//                }
//                else
//                {
//                    PrepareInputUnsafe(_bufGateIn, GateInputSize, GateDownsample);
//                    _sessionGate.RunWithBinding(_runOptions, _bindGate);
//                    res = ReadOutput(_bufGateOut);
//                }

//                // 4. Logic & Snapshot
//                StateMachine.DetectionEnabled = DetectionEnabled;

//                // Optimized Snapshot extraction into pre-pinned buffer
//                Mat snapshot = ExtractSnapshotUnsafe();

//                RippleOut output = StateMachine.Update(res.Sig, res.Art, input.BnoOk, snapshot);

//                // 5. Feedback Loop
//                bool artifactOk = res.Art < StateMachine.ArtifactThreshold;
//                bool gatesOpen = input.BnoOk && artifactOk;

//                if (!gatesOpen) _currentK = KBelowGate;
//                else if (output.State == RippleState.Possible) _currentK = KAtGate;
//                else _currentK = KBelowGate;

//                output.StrideUsed = _currentK;
//                return (RippleOut?)output;
//            })
//            .Where(o => o.HasValue)
//            .Select(o => o.Value);
//        }

//        // ==============================================================================
//        // UNSAFE MEMORY OPERATIONS
//        // ==============================================================================

//        private unsafe void UpdateRingBufferUnsafe(Mat[] mats)
//        {
//            int count = Math.Min(BatchSize, mats.Length);
//            int cols = mats[0].Cols; // Number of new time samples

//            for (int b = 0; b < count; b++)
//            {
//                float* srcPtr = (float*)mats[b].Data.ToPointer();

//                // Get pointer to pinned ring buffer
//                float* ringPtr = (float*)_hRingBuffers[b].AddrOfPinnedObject().ToPointer();

//                for (int t = 0; t < cols; t++)
//                {
//                    int ringIdx = (_headIndex + t) % RingBufferCapacity;
//                    int ringOffset = ringIdx * Channels;
//                    int srcOffset = t; // Planar src: c * cols + t

//                    for (int c = 0; c < Channels; c++)
//                    {
//                        // Interleaved Ring [T, C] <--- Planar Src [C, T]
//                        ringPtr[ringOffset + c] = srcPtr[c * cols + srcOffset];
//                    }
//                }
//            }
//            _headIndex = (_headIndex + cols) % RingBufferCapacity;
//        }

//        private unsafe void PrepareInputUnsafe(float[] dstBuf, int inputSize, int step)
//        {
//            // We can get raw pointers because the arrays are pinned in Initialise()
//            fixed (float* dstPtr = dstBuf)
//            {
//                for (int b = 0; b < BatchSize; b++)
//                {
//                    if (b >= _ringBuffers.Count) break;

//                    float* ringPtr = (float*)_hRingBuffers[b].AddrOfPinnedObject().ToPointer();
//                    int batchOffset = b * inputSize * Channels;

//                    for (int t = 0; t < inputSize; t++)
//                    {
//                        // Circular buffer indexing
//                        int stepsBack = (inputSize - 1 - t) * step;
//                        int ringIdx = _headIndex - 1 - stepsBack;

//                        // Fast modulo for negative numbers
//                        if (ringIdx < 0) ringIdx += RingBufferCapacity;
//                        // Handle edge case where step size wraps it multiple times (rare but safe)
//                        while (ringIdx < 0) ringIdx += RingBufferCapacity;

//                        int ringOffset = ringIdx * Channels;
//                        int dstOffset = batchOffset + t; // Interleaved Dest: batch + (c*inputSize) + t ?? 

//                        // WAIT: Model format is usually (Batch, Time, Channel) or (Batch, Channel, Time)?
//                        // The CreateTensor value was { Batch, InputSize, Channels } -> Interleaved.
//                        // So Dest is [B][T][C].

//                        // Correct Interleaved Layout copy:
//                        for (int c = 0; c < Channels; c++)
//                        {
//                            // dst[ b, t, c ]
//                            dstPtr[batchOffset + (t * Channels) + c] = ringPtr[ringOffset + c];
//                        }
//                    }
//                }
//            }
//        }

//        // ** Note on Layout ** : 
//        // If your model expects (Batch, Channels, Time), change the loop above.
//        // My code assumes (Batch, Time, Channels) based on your original code's Shape declaration.

//        private unsafe Mat ExtractSnapshotUnsafe()
//        {
//            // Fills the pre-allocated _snapshotBuffer and wraps it in a Mat
//            // This avoids 'new float[]' allocation.

//            // We assume Batch 0 for snapshot
//            float* ringPtr = (float*)_hRingBuffers[0].AddrOfPinnedObject().ToPointer();
//            float* dstPtr = (float*)_hSnapshotBuffer.AddrOfPinnedObject().ToPointer();

//            // Output Mat shape is (Channels, Time) -> Planar
//            for (int t = 0; t < ActiveInputSize; t++)
//            {
//                int ringIdx = _headIndex - 1 - (ActiveInputSize - 1 - t);
//                while (ringIdx < 0) ringIdx += RingBufferCapacity;

//                int ringOffset = ringIdx * Channels;

//                for (int c = 0; c < Channels; c++)
//                {
//                    // Planar Dest: [c, t]
//                    dstPtr[c * ActiveInputSize + t] = ringPtr[ringOffset + c];
//                }
//            }

//            // Create a Mat header that points to our existing buffer (Zero Data Copy)
//            // Note: This Mat must be used immediately before the next process call overwrites the buffer.
//            return new Mat(Channels, ActiveInputSize, Depth.F32, 1, _hSnapshotBuffer.AddrOfPinnedObject());
//        }

//        private InfResult ReadOutput(float[] buffer)
//        {
//            // Simple logic, usually fast enough not to need unsafe
//            int stride = buffer.Length / BatchSize;
//            float sig = buffer[stride - 1];
//            float art = (BatchSize > 1) ? buffer[(stride * 2) - 1] : 0;
//            return new InfResult { Sig = sig, Art = art };
//        }

//        public void Dispose()
//        {
//            // Release GCHandles
//            if (_hGateIn.IsAllocated) _hGateIn.Free();
//            if (_hGateOut.IsAllocated) _hGateOut.Free();
//            if (_hActiveIn.IsAllocated) _hActiveIn.Free();
//            if (_hActiveOut.IsAllocated) _hActiveOut.Free();
//            if (_hSnapshotBuffer.IsAllocated) _hSnapshotBuffer.Free();

//            if (_hRingBuffers != null)
//            {
//                foreach (var h in _hRingBuffers) if (h.IsAllocated) h.Free();
//            }

//            // Release ORT
//            _valGateIn?.Dispose(); _valGateOut?.Dispose();
//            _valActiveIn?.Dispose(); _valActiveOut?.Dispose();
//            _bindGate?.Dispose(); _bindActive?.Dispose();
//            _runOptions?.Dispose();
//            _sessionGate?.Dispose(); _sessionActive?.Dispose();
//        }
//    }
//}