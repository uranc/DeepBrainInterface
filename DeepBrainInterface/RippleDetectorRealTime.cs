using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Maximum Performance: OrtIoBinding + Pinned Memory + Latency Optimization.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorOptimized
    {
        public string ModelPath { get; set; } = "ripple_detector.onnx";

        // Input Dimensions
        public int TimePoints { get; set; } = 92;
        public int Channels { get; set; } = 8;

        // ---- ONNX RESOURCES ----
        private InferenceSession _session;
        private OrtIoBinding _binding; // UPDATED HERE
        private RunOptions _runOptions;
        private OrtMemoryInfo _memInfo;

        private string _inputName;
        private string _outputName;

        // ---- PINNED MEMORY BUFFERS ----
        private float[] _inputBuffer;
        private GCHandle _inputPin;
        private OrtValue _inputOrtValue;

        private float[] _outputBuffer;
        private GCHandle _outputPin;
        private OrtValue _outputOrtValue;

        private int _currentCapacity = 0;
        private int _activeBatchSize = 0;

        private void InitializeSession()
        {
            if (_session != null) return;

            // PERFORMANCE TUNING: OPTIMIZE FOR LATENCY
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableCpuMemArena = true
            };

            // CPU Specifics (Critical for Intel CPUs)
            options.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

            _session = new InferenceSession(ModelPath, options);
            _memInfo = OrtMemoryInfo.DefaultInstance;

            // Create the binding (using the corrected type)
            _binding = _session.CreateIoBinding();

            _runOptions = new RunOptions();

            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();
        }

        private void PrepareBinding(int batchSize)
        {
            InitializeSession();

            bool capacityChanged = batchSize > _currentCapacity;
            bool shapeChanged = batchSize != _activeBatchSize;

            // 1. Resize Physical Memory (If needed)
            if (capacityChanged)
            {
                DisposeOrtValues();
                if (_inputPin.IsAllocated) _inputPin.Free();
                if (_outputPin.IsAllocated) _outputPin.Free();

                _currentCapacity = Math.Max(batchSize, _currentCapacity);

                // Allocate Input [Batch * Time * Channels]
                int inSize = _currentCapacity * TimePoints * Channels;
                _inputBuffer = new float[inSize];
                _inputPin = GCHandle.Alloc(_inputBuffer, GCHandleType.Pinned);

                // Allocate Output [Batch * 1]
                int outSize = _currentCapacity;
                _outputBuffer = new float[outSize];
                _outputPin = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);
            }

            // 2. Update ONNX Binding (If shape changed)
            if (capacityChanged || shapeChanged)
            {
                DisposeOrtValues();

                long[] inShape = new long[] { batchSize, TimePoints, Channels };
                long[] outShape = new long[] { batchSize, 1 };

                unsafe
                {
                    // Create Zero-Copy Native Wrapper for Input
                    _inputOrtValue = OrtValue.CreateTensorValueWithData(
                        _memInfo,
                        TensorElementType.Float,
                        inShape,
                        _inputPin.AddrOfPinnedObject(),
                        batchSize * TimePoints * Channels * sizeof(float)
                    );

                    // Create Zero-Copy Native Wrapper for Output
                    _outputOrtValue = OrtValue.CreateTensorValueWithData(
                        _memInfo,
                        TensorElementType.Float,
                        outShape,
                        _outputPin.AddrOfPinnedObject(),
                        batchSize * sizeof(float)
                    );
                }

                // Bind wrappers to the session
                _binding.ClearBoundInputs();
                _binding.ClearBoundOutputs();
                _binding.BindInput(_inputName, _inputOrtValue);
                _binding.BindOutput(_outputName, _outputOrtValue);

                _activeBatchSize = batchSize;
            }
        }

        private void DisposeOrtValues()
        {
            _inputOrtValue?.Dispose();
            _inputOrtValue = null;
            _outputOrtValue?.Dispose();
            _outputOrtValue = null;
        }

        private Mat RunInference(params Mat[] mats)
        {
            int batchSize = mats.Length;
            if (batchSize == 0) return null;

            // Step 1: Ensure Memory & Bindings are correct
            PrepareBinding(batchSize);

            // Step 2: Copy Data to Pinned Buffer (Unsafe Fast Copy)
            unsafe
            {
                float* dstBase = (float*)_inputPin.AddrOfPinnedObject().ToPointer();
                int stride = TimePoints * Channels;
                long bytesPerMat = stride * sizeof(float);

                for (int i = 0; i < batchSize; i++)
                {
                    float* src = (float*)mats[i].Data.ToPointer();
                    float* dst = dstBase + (i * stride);

                    if (mats[i].Rows == TimePoints && mats[i].Cols == Channels)
                    {
                        Buffer.MemoryCopy(src, dst, bytesPerMat, bytesPerMat);
                    }
                    else if (mats[i].Rows == Channels && mats[i].Cols == TimePoints)
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

            // Step 3: EXECUTE (Zero Allocation)
            _session.RunWithBinding(_runOptions, _binding);

            // Step 4: Retrieve Result
            var outMat = new Mat(batchSize, 1, Depth.F32, 1);
            unsafe
            {
                float* resultPtr = (float*)outMat.Data.ToPointer();
                Marshal.Copy(_outputBuffer, 0, (IntPtr)resultPtr, batchSize);
            }

            return outMat;
        }

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(m => RunInference(m));
        }

        public IObservable<Mat> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return source.Select(t => RunInference(t.Item1, t.Item2));
        }

        public void Unload()
        {
            DisposeOrtValues();
            if (_inputPin.IsAllocated) _inputPin.Free();
            if (_outputPin.IsAllocated) _outputPin.Free();
            _binding?.Dispose();
            _runOptions?.Dispose();
            _session?.Dispose();
        }
    }
}