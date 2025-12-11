using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design; // Required for UITypeEditor
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("CPU Inference: Single-Threaded. Merges inputs. Strict BatchSize check.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorRealTime
    {
        // ==============================================================================
        // 1. CONFIGURATION
        // ==============================================================================
        [Category("Model")]
        [DisplayName("Model Path")]
        [Description("Path to the ONNX model file.")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", typeof(UITypeEditor))]
        [FileNameFilter("ONNX Models|*.onnx|All Files|*.*")]
        public string ModelPath { get; set; } = @"C:\DeepBrain\Models\ripple_detector.onnx";

        // ---- DIMENSIONS (PRE-ALLOCATION) ----
        [Category("Dimensions")]
        [Description("Strict Batch Size. Input Tuple size MUST match this.")]
        public int BatchSize { get; set; } = 2;

        [Category("Dimensions")]
        [Description("Time points per input sample.")]
        public int TimePoints { get; set; } = 44;

        [Category("Dimensions")]
        [Description("Channels per input sample.")]
        public int Channels { get; set; } = 8;

        // ==============================================================================
        // INTERNAL STATE
        // ==============================================================================
        private InferenceSession _session;
        private OrtIoBinding _binding;
        private RunOptions _runOpts;

        private GCHandle _hInput, _hOutput;
        private float[] _outputBuffer;

        // Caching sizes for fast copy
        private int _inputStrideFloats;
        private int _inputStrideBytes;
        private int _outputCols;

        // ==============================================================================
        // INITIALIZATION
        // ==============================================================================
        private void Initialize()
        {
            if (_session != null) return;
            if (!File.Exists(ModelPath)) throw new FileNotFoundException("Model not found", ModelPath);

            // 1. Force Single-Threaded CPU
            var opts = new SessionOptions
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
            };

            // 2. Create Session
            try
            {
                _session = new InferenceSession(ModelPath, opts);
                _binding = _session.CreateIoBinding();
                _runOpts = new RunOptions();
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to load model at {ModelPath}. {ex.Message}");
            }

            // 3. Pre-Calculate Strides
            _inputStrideFloats = TimePoints * Channels;
            _inputStrideBytes = _inputStrideFloats * sizeof(float);

            // 4. Allocate Input Buffer (Pinned)
            long[] inShape = new long[] { (long)BatchSize, (long)TimePoints, (long)Channels };
            float[] inBuffer = new float[BatchSize * _inputStrideFloats];
            _hInput = GCHandle.Alloc(inBuffer, GCHandleType.Pinned);

            // 5. Determine Output Shape (Fixing int[] -> long[] conversion)
            var outMeta = _session.OutputMetadata.First();
            int[] dimInts = outMeta.Value.Dimensions;

            long[] outShape = new long[dimInts.Length];
            for (int i = 0; i < dimInts.Length; i++)
            {
                outShape[i] = (long)dimInts[i];
            }

            // Fix dynamic batch dimension if present
            if (outShape[0] <= 0) outShape[0] = (long)BatchSize;

            // Get number of columns (e.g. 1 for prob only, 2 for prob+artifact)
            _outputCols = (int)outShape[outShape.Length - 1];

            long totalOutputFloats = 1;
            foreach (long dim in outShape) totalOutputFloats *= dim;

            _outputBuffer = new float[totalOutputFloats];
            _hOutput = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);

            // 6. Bind Buffers (Zero-Copy)
            var memInfo = OrtMemoryInfo.DefaultInstance;

            using (var inOrt = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(inBuffer), inShape))
            using (var outOrt = OrtValue.CreateTensorValueFromMemory(memInfo, new Memory<float>(_outputBuffer), outShape))
            {
                _binding.BindInput(_session.InputMetadata.Keys.First(), inOrt);
                _binding.BindOutput(_session.OutputMetadata.Keys.First(), outOrt);
            }

            // Warmup
            _session.RunWithBinding(_runOpts, _binding);
        }

        // ==============================================================================
        // CORE PROCESSING
        // ==============================================================================
        private Mat ProcessInputs(params Mat[] inputs)
        {
            Initialize();

            // STRICT VALIDATION
            if (inputs.Length != BatchSize)
            {
                throw new InvalidOperationException($"Input count ({inputs.Length}) does not match configured BatchSize ({BatchSize}).");
            }

            unsafe
            {
                float* dstBase = (float*)_hInput.AddrOfPinnedObject();

                // 1. LOOP & MERGE
                for (int i = 0; i < inputs.Length; i++)
                {
                    Mat m = inputs[i];
                    float* src = (float*)m.Data.ToPointer();

                    // Direct Memory Copy
                    Buffer.MemoryCopy(src, dstBase + (i * _inputStrideFloats), _inputStrideBytes, _inputStrideBytes);
                }
            }

            // 2. RUN INFERENCE
            _session.RunWithBinding(_runOpts, _binding);

            // 3. RETURN OUTPUT
            // Rows = BatchSize, Cols = OutputCols.
            var resultMat = new Mat(BatchSize, _outputCols, Depth.F32, 1);
            Marshal.Copy(_outputBuffer, 0, resultMat.Data, _outputBuffer.Length);

            return resultMat;
        }

        // ==============================================================================
        // OVERLOADS (2, 3, 4 Inputs)
        // ==============================================================================

        // 1. Single Input
        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(m => ProcessInputs(m));
        }

        // 2. Pair
        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return source.Select(t => ProcessInputs(t.Item1, t.Item2));
        }

        // 3. Triplet
        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat, Mat>> source)
        {
            return source.Select(t => ProcessInputs(t.Item1, t.Item2, t.Item3));
        }

        // 4. Quad
        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat, Mat, Mat>> source)
        {
            return source.Select(t => ProcessInputs(t.Item1, t.Item2, t.Item3, t.Item4));
        }

        // Cleanup
        public void Dispose()
        {
            if (_hInput.IsAllocated) _hInput.Free();
            if (_hOutput.IsAllocated) _hOutput.Free();
            _binding?.Dispose();
            _session?.Dispose();
        }
    }
}