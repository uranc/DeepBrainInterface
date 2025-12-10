using Bonsai;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCV.Net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    public enum OnnxProvider { Cpu, Cuda, TensorRT }

    [Combinator]
    [Description("High-Performance GPU Inference. Returns a pair: (Prediction Mat, Duration in ms).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorGPU
    {
        /* ───── User Parameters ─────────────────────────────────────────── */
        public string ModelPath { get; set; } = @"ripple_detector.onnx";
        public OnnxProvider Provider { get; set; } = OnnxProvider.Cuda;
        public int DeviceId { get; set; } = 0;
        public int TimePoints { get; set; } = 92;
        public int Channels { get; set; } = 8;
        /* ───────────────────────────────────────────────────────────────── */

        // Resources
        InferenceSession _session;
        OrtIoBinding _binding;
        RunOptions _runOptions;
        OrtMemoryInfo _memInfo;
        Stopwatch _timer = new Stopwatch();

        // Buffers
        float[] _inputBuffer; GCHandle _inputPin; OrtValue _inputOrtValue;
        float[] _outputBuffer; GCHandle _outputPin; OrtValue _outputOrtValue;
        int _currentCapacity = 0;
        int _activeBatchSize = 0;
        string _inputName, _outputName;

        // [Standard Session Initialization - identical to previous versions]
        private void InitializeSession()
        {
            if (_session != null) return;
            var opts = new SessionOptions();
            try
            {
                if (Provider == OnnxProvider.TensorRT)
                {
                    Environment.SetEnvironmentVariable("ORT_TENSORRT_FP16_ENABLE", "1");
                    opts.AppendExecutionProvider_Tensorrt(DeviceId);
                    opts.AppendExecutionProvider_CUDA(DeviceId);
                }
                else if (Provider == OnnxProvider.Cuda) opts.AppendExecutionProvider_CUDA(DeviceId);
            }
            catch { Provider = OnnxProvider.Cpu; }

            _session = new InferenceSession(ModelPath, opts);
            _memInfo = OrtMemoryInfo.DefaultInstance;
            _binding = _session.CreateIoBinding();
            _runOptions = new RunOptions();
            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();
        }

        private void PrepareBinding(int batchSize)
        {
            InitializeSession();
            if (batchSize > _currentCapacity)
            {
                if (_inputPin.IsAllocated) _inputPin.Free();
                if (_outputPin.IsAllocated) _outputPin.Free();
                _currentCapacity = Math.Max(batchSize, _currentCapacity);
                _inputBuffer = new float[_currentCapacity * TimePoints * Channels];
                _inputPin = GCHandle.Alloc(_inputBuffer, GCHandleType.Pinned);
                _outputBuffer = new float[_currentCapacity];
                _outputPin = GCHandle.Alloc(_outputBuffer, GCHandleType.Pinned);
            }
            if (batchSize != _activeBatchSize || batchSize > _currentCapacity)
            {
                _inputOrtValue?.Dispose(); _outputOrtValue?.Dispose();
                unsafe
                {
                    long[] inShape = { batchSize, TimePoints, Channels };
                    long[] outShape = { batchSize, 1 };
                    _inputOrtValue = OrtValue.CreateTensorValueWithData(_memInfo, TensorElementType.Float, inShape, _inputPin.AddrOfPinnedObject(), batchSize * TimePoints * Channels * sizeof(float));
                    _outputOrtValue = OrtValue.CreateTensorValueWithData(_memInfo, TensorElementType.Float, outShape, _outputPin.AddrOfPinnedObject(), batchSize * sizeof(float));
                }
                _binding.ClearBoundInputs(); _binding.BindInput(_inputName, _inputOrtValue);
                _binding.ClearBoundOutputs(); _binding.BindOutput(_outputName, _outputOrtValue);
                _activeBatchSize = batchSize;
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Modified: Returns Tuple<Mat, double>
        // ─────────────────────────────────────────────────────────────────
        private Tuple<Mat, double> RunInference(IList<Mat> mats)
        {
            int batchSize = mats.Count;
            if (batchSize == 0) return null;

            PrepareBinding(batchSize);

            // Copy Data
            unsafe
            {
                float* dstBase = (float*)_inputPin.AddrOfPinnedObject().ToPointer();
                int stride = TimePoints * Channels;
                long bytesPerMat = stride * sizeof(float);
                for (int i = 0; i < batchSize; i++)
                {
                    float* src = (float*)mats[i].Data.ToPointer();
                    Buffer.MemoryCopy(src, dstBase + (i * stride), bytesPerMat, bytesPerMat);
                }
            }

            // 1. Start Timer
            _timer.Restart();

            // 2. Run Inference
            _session.RunWithBinding(_runOptions, _binding);

            // 3. Stop Timer
            _timer.Stop();
            double duration = _timer.Elapsed.TotalMilliseconds;

            // 4. Create Output
            var outMat = new Mat(batchSize, 1, Depth.F32, 1);
            unsafe
            {
                float* resultPtr = (float*)outMat.Data.ToPointer();
                Marshal.Copy(_outputBuffer, 0, (IntPtr)resultPtr, batchSize);
            }

            // 5. Return Pair (Item1 = Matrix, Item2 = Duration)
            return Tuple.Create(outMat, duration);
        }

        // Updated Overloads
        public IObservable<Tuple<Mat, double>> Process(IObservable<Mat> source)
        {
            return source.Select(m => RunInference(new[] { m }));
        }

        public IObservable<Tuple<Mat, double>> Process(IObservable<IList<Mat>> source)
        {
            return source.Select(batch => RunInference(batch));
        }

        public void Unload()
        {
            _inputOrtValue?.Dispose(); _outputOrtValue?.Dispose();
            if (_inputPin.IsAllocated) _inputPin.Free();
            if (_outputPin.IsAllocated) _outputPin.Free();
            _binding?.Dispose(); _session?.Dispose();
        }
    }
}