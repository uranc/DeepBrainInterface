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
        // INTERNAL RESOURCES
        // ==============================================================================
        InferenceSession _session;
        OrtIoBinding _ioBinding;
        RunOptions _runOptions;
        OrtMemoryInfo _memInfo;

        // Fixed Pinned Memory
        float[] _inputBuffer;
        GCHandle _inputPin;
        OrtValue _inputOrtValue;

        float[] _outputBuffer;
        GCHandle _outputPin;
        OrtValue _outputOrtValue;

        struct InputPackage { public Mat[] Mats; }

        private void Initialise()
        {
            if (_session != null) return;
            if (!File.Exists(ModelPath)) throw new FileNotFoundException("Model not found", ModelPath);

            // A. CONFIGURE SESSION
            var opts = new SessionOptions
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                EnableCpuMemArena = true
            };
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

            _session = new InferenceSession(ModelPath, opts);
            _memInfo = OrtMemoryInfo.DefaultInstance;
            _ioBinding = _session.CreateIoBinding();
            _runOptions = new RunOptions();

            // B. STRICT ALLOCATION
            int inputLen = BatchSize * TimePoints * Channels;
            _inputBuffer = new float[inputLen];
            _inputPin = GCHandle.Alloc(_inputBuffer, GCHandleType.Pinned);

            int outputLen = BatchSize; // [Batch, 1]
            _outputBuffer = new float[outputLen];
            _outputPin = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);

            // C. BINDING SETUP
            var inputName = _session.InputMetadata.Keys.First();
            var outputName = _session.OutputMetadata.Keys.First();

            long[] inShape = new long[] { BatchSize, TimePoints, Channels };
            long[] outShape = new long[] { BatchSize, 1 };

            unsafe
            {
                _inputOrtValue = OrtValue.CreateTensorValueWithData(
                    _memInfo,
                    TensorElementType.Float,
                    inShape,
                    _inputPin.AddrOfPinnedObject(),
                    inputLen * sizeof(float)
                );

                _outputOrtValue = OrtValue.CreateTensorValueWithData(
                    _memInfo,
                    TensorElementType.Float,
                    outShape,
                    _outputPin.AddrOfPinnedObject(),
                    outputLen * sizeof(float)
                );
            }

            _ioBinding.BindInput(inputName, _inputOrtValue);
            _ioBinding.BindOutput(outputName, _outputOrtValue);

            // Warmup
            _session.RunWithBinding(_runOptions, _ioBinding);
        }

        private IObservable<Mat> ProcessInternal(IObservable<InputPackage> source)
        {
            return source.Select(input =>
            {
                Initialise();

                if (input.Mats.Length != BatchSize)
                    throw new InvalidOperationException($"Input count {input.Mats.Length} does not match BatchSize {BatchSize}");

                // 1. FAST COPY
                unsafe
                {
                    float* dstBase = (float*)_inputPin.AddrOfPinnedObject().ToPointer();
                    int stride = TimePoints * Channels;
                    long bytesPerMat = stride * sizeof(float);

                    for (int i = 0; i < BatchSize; i++)
                    {
                        float* src = (float*)input.Mats[i].Data.ToPointer();
                        float* dst = dstBase + (i * stride);

                        if (input.Mats[i].Rows == TimePoints && input.Mats[i].Cols == Channels)
                        {
                            Buffer.MemoryCopy(src, dst, bytesPerMat, bytesPerMat);
                        }
                        else if (input.Mats[i].Rows == Channels && input.Mats[i].Cols == TimePoints)
                        {
                            for (int c = 0; c < Channels; c++)
                            {
                                int cOff = c * TimePoints;
                                for (int t = 0; t < TimePoints; t++)
                                {
                                    dst[t * Channels + c] = src[cOff + t];
                                }
                            }
                        }
                        else
                        {
                            Buffer.MemoryCopy(src, dst, bytesPerMat, bytesPerMat);
                        }
                    }
                }

                // 2. INFERENCE
                _session.RunWithBinding(_runOptions, _ioBinding);

                // 3. RETURN RESULT (Copy Pinned Output -> New Mat)
                var outMat = new Mat(BatchSize, 1, Depth.F32, 1);
                unsafe
                {
                    float* dst = (float*)outMat.Data.ToPointer();
                    Marshal.Copy(_outputBuffer, 0, (IntPtr)dst, BatchSize);
                }

                return outMat;
            });
        }

        // --- PIPELINE OVERLOADS ---

        public IObservable<Mat> Process(IObservable<Mat> source)
            => ProcessInternal(source.Select(m => new InputPackage { Mats = new[] { m } }));

        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
            => ProcessInternal(source.Select(t => new InputPackage { Mats = new[] { t.Item1, t.Item2 } }));

        public void Unload()
        {
            _inputOrtValue?.Dispose(); _outputOrtValue?.Dispose();
            if (_inputPin.IsAllocated) _inputPin.Free();
            if (_outputPin.IsAllocated) _outputPin.Free();
            _ioBinding?.Dispose(); _runOptions?.Dispose(); _session?.Dispose();
        }
    }
}