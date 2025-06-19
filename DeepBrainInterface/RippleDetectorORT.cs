//using Bonsai;
//using Microsoft.ML.OnnxRuntime;
//using Microsoft.ML.OnnxRuntime.Tensors;
//using OpenCV.Net;
//using System;
//using System.Collections.Generic;
//using System.ComponentModel;
//using System.Drawing.Design;
//using System.Linq;
//using System.Reactive.Concurrency;
//using System.Reactive.Linq;
//using System.Runtime.InteropServices;
//using System.Threading;
//using System.Windows.Forms.Design;

//namespace DeepBrainInterface
//{
//    public enum OrTProvider { Cpu, Cuda, TensorRT }

//    [Combinator]
//    [Description("Runs an ONNX model on streaming Mat data via ORT, one inference per frame. Drops frames if still busy.")]
//    [WorkflowElementCategory(ElementCategory.Transform)]
//    public class RippleDetectorORT : IDisposable // Implement IDisposable for proper cleanup
//    {
//        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
//        [Description("Path to the .onnx file.")]
//        public string ModelPath { get; set; } = @"C:\Models\ripple_detector.onnx";

//        [Description("Execution provider (Cpu, Cuda, TensorRT).")]
//        public OrTProvider Provider { get; set; } = OrTProvider.Cpu;

//        [Description("When Provider ≠ Cpu, GPU device index.")]
//        public int DeviceId { get; set; } = 0;

//        [Description("Graph input name (blank = auto).")]
//        public string InputName { get; set; } = "";

//        [Description("Graph output name (blank = auto).")]
//        public string OutputName { get; set; } = "";

//        [Description("Time dimension length.")]
//        public int ExpectedTimepoints { get; set; } = 1104;

//        [Description("Channel count.")]
//        public int ExpectedChannels { get; set; } = 8;

//        InferenceSession _session;
//        string _actualInputName, _actualOutputName; // Renamed to avoid confusion with properties
//        float[] _scratchBuffer; // Renamed for clarity
//        readonly List<NamedOnnxValue> _feed = new List<NamedOnnxValue>(1);
//        int _busy = 0; // 0 = idle, 1 = inference in flight
//        private bool _disposedValue;

//        void InitializeSession()
//        {
//            if (_session != null) return;

//            SessionOptions opts = null;
//            try
//            {
//                opts = new SessionOptions();
//                switch (Provider)
//                {
//                    case OrTProvider.Cuda:
//                        opts.AppendExecutionProvider_CUDA(DeviceId);
//                        break;
//                    case OrTProvider.TensorRT:
//                        // Order can matter: TensorRT often benefits from being tried first.
//                        // Some configurations might require specific TensorRT settings (e.g., engine cache path)
//                        opts.AppendExecutionProvider_Tensorrt(DeviceId);
//                        opts.AppendExecutionProvider_CUDA(DeviceId); // CUDA is a dependency for TensorRT execution provider
//                        break;
//                    case OrTProvider.Cpu:
//                    default:
//                        // Default is CPU, no explicit provider needed unless optimizing (e.g., OpenVINO, DNNL)
//                        break;
//                }
//                _session = new InferenceSession(ModelPath, opts);
//                _actualInputName = string.IsNullOrWhiteSpace(InputName)
//                    ? _session.InputMetadata.Keys.First()
//                    : InputName;
//                _actualOutputName = string.IsNullOrWhiteSpace(OutputName)
//                    ? _session.OutputMetadata.Keys.First()
//                    : OutputName;
//            }
//            finally
//            {
//                opts?.Dispose(); // Dispose SessionOptions whether session creation succeeded or failed
//            }
//        }

//        DenseTensor<float> ToTensor(Mat m)
//        {
//            int batchSize = 1;
//            int totalElements = batchSize * ExpectedTimepoints * ExpectedChannels;

//            if (m == null)
//            {
//                throw new ArgumentNullException(nameof(m), "Input Mat cannot be null.");
//            }
//            if (m.Depth != Depth.F32) // Ensure Mat is of type float
//            {
//                throw new ArgumentException($"Input Mat must be of type F32. Actual type: {m.Depth}.", nameof(m));
//            }


//            if (_scratchBuffer == null || _scratchBuffer.Length < totalElements)
//            {
//                _scratchBuffer = new float[totalElements];
//            }

//            unsafe
//            {
//                float* srcPtr = (float*)m.Data.ToPointer();
//                fixed (float* dstFixedPtr = &_scratchBuffer[0])
//                {
//                    float* dstPtr = dstFixedPtr;

//                    // Case 1: Input Mat is [Channels, Timepoints] (e.g., m.Rows = ExpectedChannels, m.Cols = ExpectedTimepoints)
//                    // Needs transpose to model's expected [Timepoints, Channels]
//                    if (m.Rows == ExpectedChannels && m.Cols == ExpectedTimepoints)
//                    {
//                        for (int t = 0; t < ExpectedTimepoints; t++)
//                        {
//                            for (int c = 0; c < ExpectedChannels; c++)
//                            {
//                                dstPtr[t * ExpectedChannels + c] = srcPtr[c * ExpectedTimepoints + t];
//                            }
//                        }
//                    }
//                    // Case 2: Input Mat is already [Timepoints, Channels] (e.g., m.Rows = ExpectedTimepoints, m.Cols = ExpectedChannels)
//                    else if (m.Rows == ExpectedTimepoints && m.Cols == ExpectedChannels)
//                    {
//                        int bytesToCopy = totalElements * sizeof(float);
//                        Buffer.MemoryCopy(srcPtr, dstPtr, bytesToCopy, bytesToCopy);
//                    }
//                    else
//                    {
//                        throw new ArgumentException(
//                            $"Input Mat dimensions ({m.Rows}x{m.Cols}) are incompatible. Expected either [{ExpectedChannels}x{ExpectedTimepoints}] (for transpose) or [{ExpectedTimepoints}x{ExpectedChannels}] (direct copy).", nameof(m));
//                    }
//                }
//            }
//            return new DenseTensor<float>(_scratchBuffer, new[] { batchSize, ExpectedTimepoints, ExpectedChannels });
//        }

//        public IObservable<Mat> Process(IObservable<Mat> source)
//        {
//            return source.SelectMany(m =>
//            {
//                if (Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
//                {
//                    // System.Diagnostics.Debug.WriteLine("RippleDetectorORT: Dropped frame, processor busy.");
//                    return Observable.Empty<Mat>();
//                }

//                return Observable.Start(() =>
//                {
//                    NamedOnnxValue onnxInputValue = null;
//                    Mat resultMat = null;
//                    try
//                    {
//                        if (_disposedValue) // Prevent operation if disposed
//                        {
//                            throw new ObjectDisposedException(nameof(RippleDetectorORT));
//                        }

//                        InitializeSession(); // Idempotent initialization

//                        var inputTensor = ToTensor(m); // Can throw ArgumentException or ArgumentNullException

//                        _feed.Clear(); // Clear previous inputs
//                        onnxInputValue = NamedOnnxValue.CreateFromTensor(_actualInputName, inputTensor);
//                        _feed.Add(onnxInputValue);

//                        using (var results = _session.Run(_feed)) // Core inference, can throw
//                        {
//                            var outputValue = results.FirstOrDefault(val => val.Name == _actualOutputName);
//                            if (outputValue == null)
//                            {
//                                throw new InvalidOperationException($"Output tensor with name '{_actualOutputName}' not found in model results. Available outputs: {string.Join(", ", results.Select(r => r.Name))}");
//                            }

//                            var outputTensor = outputValue.AsTensor<float>();
//                            if (outputTensor.Length == 0)
//                            {
//                                throw new InvalidOperationException("Output tensor is empty.");
//                            }
//                            float score = outputTensor.GetValue(0); // Assuming single scalar output

//                            resultMat = new Mat(1, 1, Depth.F32, 1); // Create Mat for the single float score
//                            MarshalExtensions.WriteFloat(resultMat.Data, score);
//                        }
//                        return resultMat;
//                    }
//                    // No catch block here: Let exceptions propagate to the .Catch operator below.
//                    // This ensures that Rx's error handling mechanisms are used.
//                    finally
//                    {
//                        // Dispose of the created NamedOnnxValue if it's IDisposable
//                        (onnxInputValue as IDisposable)?.Dispose();
//                        Interlocked.Exchange(ref _busy, 0); // Always mark as not busy
//                    }
//                }, Scheduler.Default) // Offload to a ThreadPool thread
//                .Catch<Mat, Exception>(ex => // Catch exceptions from the Observable.Start task or upstream
//                {
//                    // Log the exception. In a Bonsai context, this error will propagate downstream.
//                    // If this is the end of a branch, Bonsai might show it in the output or log.
//                    System.Diagnostics.Trace.TraceError($"Error in RippleDetectorORT processing: {ex}");

//                    // Ensure _busy is reset if an exception occurred before the finally block in the task
//                    // (though the current structure makes that unlikely for the task's code itself).
//                    Interlocked.Exchange(ref _busy, 0);

//                    return Observable.Throw<Mat>(ex); // Propagate the error through the observable sequence
//                });
//            });
//        }

//        protected virtual void Dispose(bool disposing)
//        {
//            if (!_disposedValue)
//            {
//                if (disposing)
//                {
//                    // Dispose managed state (managed objects).
//                    _session?.Dispose();
//                    _session = null;
//                }
//                _disposedValue = true;
//            }
//        }

//        public void Dispose()
//        {
//            Dispose(disposing: true);
//            GC.SuppressFinalize(this);
//        }
//    }

//    public static class MarshalExtensions
//    {
//        public static void WriteFloat(IntPtr ptr, float value)
//        {
//            byte[] bytes = BitConverter.GetBytes(value);
//            Marshal.Copy(bytes, 0, ptr, bytes.Length);
//        }
//    }
//}