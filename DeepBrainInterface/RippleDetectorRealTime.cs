using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Drawing.Design;
using System.Windows.Forms.Design;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("High-Performance ONNX Inference (Strict Batch 1 or 2, Pinned Memory).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorRealTime
    {
        // ==============================================================================
        // 1. CONFIGURATION
        // ==============================================================================
        [Category("Model")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("Model")]
        [Description("Strictly 1 (Signal) or 2 (Signal + Artifact).")]
        public int BatchSize { get; set; } = 2;

        [Category("Model")]
        public int TimePoints { get; set; } = 44;

        [Category("Model")]
        public int Channels { get; set; } = 8;

        // ==============================================================================
        // 2. INTERNAL RESOURCES
        // ==============================================================================
        private InferenceSession _session;
        private OrtIoBinding _ioBinding;
        private RunOptions _runOptions;

        // Fixed Pinned Memory (Prevent GC movement)
        private GCHandle _inputPin;
        private GCHandle _outputPin;
        private OrtValue _inputOrtValue;
        private OrtValue _outputOrtValue;

        // Buffers
        private float[] _outputBuffer; // Managed array for result copying
        private int _batchStrideFloats; // Floats per batch item

        private void Initialise()
        {
            if (_session != null) return;
            if (!File.Exists(ModelPath)) throw new FileNotFoundException("Model not found", ModelPath);

            // A. PERFORMANCE TWEAKS & SESSION CONFIG
            var opts = new SessionOptions
            {
                // Critical: Force single-threaded execution to prevent overhead
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                EnableCpuMemArena = true
            };

            // Critical: Treat tiny floats (denormals) as zero. 
            // Prevents massive CPU spikes on silence/baseline signals.
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

            // Critical: Stop ONNX from busy-waiting (spinning) on the CPU
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");

            _session = new InferenceSession(ModelPath, opts);
            _ioBinding = _session.CreateIoBinding();
            _runOptions = new RunOptions();

            // B. SIZE CALCULATION
            // Model Expects: [Batch, Time, Channels] -> [Batch, 44, 8]
            // Input comes as: [Channels, Time] -> [8, 44] (Must Transpose)
            _batchStrideFloats = TimePoints * Channels;
            int totalInputFloats = BatchSize * _batchStrideFloats;
            int totalOutputFloats = BatchSize; // [Batch, 1]

            // C. PINNED ALLOCATION (Zero Copy Lifecycle)
            var inputData = new float[totalInputFloats];
            _inputPin = GCHandle.Alloc(inputData, GCHandleType.Pinned);

            _outputBuffer = new float[totalOutputFloats];
            _outputPin = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);

            // D. BINDING (Zero Copy Inference)
            var memInfo = OrtMemoryInfo.DefaultInstance;
            unsafe
            {
                // Bind Input
                _inputOrtValue = OrtValue.CreateTensorValueWithData(
                    memInfo, TensorElementType.Float,
                    new long[] { BatchSize, TimePoints, Channels },
                    _inputPin.AddrOfPinnedObject(),
                    totalInputFloats * sizeof(float)
                );

                // Bind Output
                _outputOrtValue = OrtValue.CreateTensorValueWithData(
                    memInfo, TensorElementType.Float,
                    new long[] { BatchSize, 1 },
                    _outputPin.AddrOfPinnedObject(),
                    _outputBuffer.Length * sizeof(float)
                );
            }

            _ioBinding.BindInput(_session.InputMetadata.Keys.First(), _inputOrtValue);
            _ioBinding.BindOutput(_session.OutputMetadata.Keys.First(), _outputOrtValue);

            // Warmup
            _session.RunWithBinding(_runOptions, _ioBinding);
        }

        // ==============================================================================
        // 3. PROCESSING (High Performance Overloads)
        // ==============================================================================

        // Scenario: Batch Size 2 (Signal + Artifact)
        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return source.Select(input =>
            {
                Initialise();
                if (BatchSize != 2) throw new InvalidOperationException("Model Configured for Batch 2, but received Tuple.");

                unsafe
                {
                    float* ptr = (float*)_inputPin.AddrOfPinnedObject();

                    // 1. Transpose First Mat (Signal) -> Buffer Index 0
                    RobustTransposeCopy(input.Item1, ptr);

                    // 2. Transpose Second Mat (Artifact) -> Buffer Index 352 (44*8)
                    RobustTransposeCopy(input.Item2, ptr + _batchStrideFloats);
                }

                return RunInference();
            });
        }

        // Scenario: Batch Size 1 (Signal Only)
        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                Initialise();
                if (BatchSize != 1) throw new InvalidOperationException("Model Configured for Batch 1, but received Single Mat.");

                unsafe
                {
                    // Transpose Mat -> Buffer Index 0
                    RobustTransposeCopy(input, (float*)_inputPin.AddrOfPinnedObject());
                }

                return RunInference();
            });
        }

        // ==============================================================================
        // 4. CORE LOGIC (Robust Memory Ops)
        // ==============================================================================

        // Converts 8x44 (Input) -> 44x8 (Model) safely handling padding/ROIs
        private unsafe void RobustTransposeCopy(Mat src, float* dstBase)
        {
            // Strict Dimensions Check
            if (src.Rows != Channels || src.Cols != TimePoints)
                throw new InvalidOperationException($"Input Mat must be {Channels}x{TimePoints}. Got {src.Rows}x{src.Cols}");

            byte* srcRowPtr = (byte*)src.Data.ToPointer();
            int srcStep = src.Step; // Stride in bytes (handles padding/ROIs)

            // Loop Input Rows (Channels = 8)
            for (int c = 0; c < Channels; c++)
            {
                float* srcFloatRow = (float*)srcRowPtr;

                // Loop Input Cols (Time = 44)
                for (int t = 0; t < TimePoints; t++)
                {
                    // Transpose mapping:
                    // Input:  [Channel, Time]
                    // Output: [Time, Channel] (Linear Index: t * 8 + c)
                    dstBase[t * Channels + c] = srcFloatRow[t];
                }

                // Advance to next row safely using Step
                srcRowPtr += srcStep;
            }
        }

        private Mat RunInference()
        {
            // 1. Execute (Zero-Copy)
            _session.RunWithBinding(_runOptions, _ioBinding);

            // 2. Result (Minimal Allocation)
            // We create a tiny 1x2 Mat for the output. 
            // This is the only per-frame allocation.
            var outMat = new Mat(BatchSize, 1, Depth.F32, 1);
            unsafe
            {
                Marshal.Copy(_outputBuffer, 0, outMat.Data, BatchSize);
            }
            return outMat;
        }

        public void Unload()
        {
            _inputOrtValue?.Dispose();
            _outputOrtValue?.Dispose();
            if (_inputPin.IsAllocated) _inputPin.Free();
            if (_outputPin.IsAllocated) _outputPin.Free();
            _ioBinding?.Dispose();
            _runOptions?.Dispose();
            _session?.Dispose();
            _session = null;
        }
    }
}