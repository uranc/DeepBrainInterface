using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
// We remove the ambiguity by not using 'using System.Diagnostics;' for the class name
// We will type the full name below.

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Adaptive Ripple Detector: High Priority. Zero-Copy Mode.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RippleDetectorAdaptive
    {
        // --- CONFIGURATION ---

        [Category("Model")]
        [Description("Path to the ONNX model file.")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", "System.Drawing.Design.UITypeEditor, System.Drawing")]
        [FileNameFilter("ONNX Models|*.onnx|All Files|*.*")]
        public string ModelPath { get; set; } = @"ripple_detector.onnx";

        [Category("Model")] public int TimePoints { get; set; } = 44;
        [Category("Model")] public int Channels { get; set; } = 8;

        [Category("Logic")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public RippleStateMachineMatBool StateMachine { get; set; } = new RippleStateMachineMatBool();

        [Category("Stride")] public int KBelowGate { get; set; } = 5;
        [Category("Stride")] public int KAtGate { get; set; } = 1;

        // --- PROCESS METHOD ---

        // Matches Workflow: Tuple< Tuple<Signal, Artifact>, BnoFlag >
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
        {
            return Observable.Using(
                () => new AdaptiveEngine(ModelPath, Channels, TimePoints),
                (AdaptiveEngine engine) =>
                    source.Select(input =>
                    {
                        var mats = input.Item1;
                        var bnoOk = input.Item2;

                        // Pass to Engine -> Calls StateMachine -> Returns RippleOut
                        return engine.Execute(mats.Item1, mats.Item2, bnoOk, StateMachine, KBelowGate, KAtGate);
                    })
            );
        }

        // --- ENGINE CLASS ---
        private class AdaptiveEngine : IDisposable
        {
            private InferenceSession _session;
            private OrtIoBinding _binding;
            private RunOptions _runOpts;

            private GCHandle _inPin, _outPin;
            private float[] _outBuffer;
            private int _strideFloats;
            private int _channels, _time;

            // Stride State
            private int _strideCounter = 0;
            private int _currentK = 1;

            public AdaptiveEngine(string path, int channels, int time)
            {
                // FIX: Use Fully Qualified Name to avoid collision with 'Process' method
                try
                {
                    System.Diagnostics.Process.GetCurrentProcess().PriorityClass = System.Diagnostics.ProcessPriorityClass.High;
                }
                catch { /* Ignore if permissions fail */ }

                _time = time;
                _channels = channels;
                _strideFloats = time * channels;

                int batch = 2;
                _outBuffer = new float[batch];

                try
                {
                    var opts = new SessionOptions();
                    opts.IntraOpNumThreads = 1;
                    opts.InterOpNumThreads = 1;
                    opts.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

                    _session = new InferenceSession(path, opts);
                    _binding = _session.CreateIoBinding();
                    _runOpts = new RunOptions();

                    var inputBuffer = new float[batch * _strideFloats];
                    _inPin = GCHandle.Alloc(inputBuffer, GCHandleType.Pinned);
                    _outPin = GCHandle.Alloc(_outBuffer, GCHandleType.Pinned);

                    var mem = OrtMemoryInfo.DefaultInstance;

                    using (var inOrt = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(inputBuffer), new long[] { batch, time, channels }))
                    using (var outOrt = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(_outBuffer), new long[] { batch, 1 }))
                    {
                        _binding.BindInput(_session.InputMetadata.Keys.First(), inOrt);
                        _binding.BindOutput(_session.OutputMetadata.Keys.First(), outOrt);
                    }
                }
                catch (Exception ex) { Console.WriteLine($"[FATAL] {ex.Message}"); throw; }
            }

            public RippleOut Execute(Mat sig, Mat art, bool bnoOk, RippleStateMachineMatBool fsm, int kBelow, int kAt)
            {
                // A. Stride Logic (Gate)
                if (_currentK > 1)
                {
                    _strideCounter++;
                    if (_strideCounter < _currentK)
                    {
                        // SKIPPED FRAME: Return dummy result
                        return new RippleOut
                        {
                            State = RippleState.NoRipple,
                            StrideUsed = _currentK,
                            SignalData = sig // Pass ref to keep visualization alive
                        };
                    }
                    _strideCounter = 0;
                }

                // B. Data Copy (Fast / Unsafe)
                unsafe
                {
                    float* ptr = (float*)_inPin.AddrOfPinnedObject();
                    CopyMat(sig, ptr);
                    CopyMat(art, ptr + _strideFloats);
                }

                // C. Inference
                _session.RunWithBinding(_runOpts, _binding);

                float sigP = _outBuffer[0];
                float artP = _outBuffer[1];

                // D. Update State Machine
                // PASS BY REF (sig) - DO NOT CLONE
                var result = fsm.Update(sigP, artP, bnoOk, sig);

                // E. Adaptive Logic
                _currentK = (result.State != RippleState.NoRipple) ? kAt : kBelow;
                result.StrideUsed = _currentK;

                return result;
            }

            private unsafe void CopyMat(Mat src, float* dst)
            {
                float* srcPtr = (float*)src.Data.ToPointer();
                int srcStep = src.Step / sizeof(float);

                int cMax = _channels;
                int tMax = _time;
                int dstIdx = 0;

                for (int t = 0; t < tMax; t++)
                {
                    for (int c = 0; c < cMax; c++)
                    {
                        dst[dstIdx++] = srcPtr[c * srcStep + t];
                    }
                }
            }

            public void Dispose()
            {
                _session?.Dispose();
                _binding?.Dispose();
                _runOpts?.Dispose();
                if (_inPin.IsAllocated) _inPin.Free();
                if (_outPin.IsAllocated) _outPin.Free();
            }
        }
    }
}