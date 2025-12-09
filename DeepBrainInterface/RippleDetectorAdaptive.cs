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
using System.Runtime;
using System.Runtime.InteropServices;
// Alias for Process to avoid collisions
using SysProcess = System.Diagnostics.Process;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Adaptive Ripple Detector: Batch (1 or 2) + Stride Gate + Logic.")]
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

        // Hardware Config
        public enum InferenceProvider { Cpu, Cuda, TensorRt }
        [Category("Hardware")] public InferenceProvider Provider { get; set; } = InferenceProvider.Cuda;
        [Category("Hardware")] public int TargetCoreIndex { get; set; } = 4;


        // --- PROCESS METHODS (Infers Batch Size) ---

        // Case 1: Batch = 2 (Signal + Artifact)
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            return Observable.Using(
                () => CreateEngine(batchSize: 2),
                (AdaptiveEngine engine) =>
                    source.Where(_ => engine.CheckStride())
                          .Select(input => engine.ExecuteBatch2(input.Item1, input.Item2, true, StateMachine, KBelowGate, KAtGate))
            );
        }

        // Case 2: Batch = 2 (Signal + Artifact + Boolean Flag)
        public IObservable<RippleOut> Process(IObservable<Tuple<Tuple<Mat, Mat>, bool>> source)
        {
            return Observable.Using(
                () => CreateEngine(batchSize: 2),
                (AdaptiveEngine engine) =>
                    source.Where(_ => engine.CheckStride())
                          .Select(input => engine.ExecuteBatch2(input.Item1.Item1, input.Item1.Item2, input.Item2, StateMachine, KBelowGate, KAtGate))
            );
        }

        // Case 3: Batch = 1 (Signal Only)
        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            return Observable.Using(
                () => CreateEngine(batchSize: 1),
                (AdaptiveEngine engine) =>
                    source.Where(_ => engine.CheckStride())
                          .Select(input => engine.ExecuteBatch1(input, true, StateMachine, KBelowGate, KAtGate))
            );
        }

        // Case 4: Batch = 1 (Signal + Boolean Flag)
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
        {
            return Observable.Using(
                () => CreateEngine(batchSize: 1),
                (AdaptiveEngine engine) =>
                    source.Where(_ => engine.CheckStride())
                          .Select(input => engine.ExecuteBatch1(input.Item1, input.Item2, StateMachine, KBelowGate, KAtGate))
            );
        }

        // --- FACTORY ---
        private AdaptiveEngine CreateEngine(int batchSize)
        {
            // Apply CPU Locking
            OptimizeThread(TargetCoreIndex);

            return new AdaptiveEngine(ModelPath, Provider, batchSize, Channels, TimePoints);
        }

        private static void OptimizeThread(int coreIndex)
        {
            try
            {
                if (Environment.OSVersion.Platform == PlatformID.Win32NT)
                {
                    SysProcess.GetCurrentProcess().ProcessorAffinity = new IntPtr(1 << coreIndex);
                    SysProcess.GetCurrentProcess().PriorityClass = ProcessPriorityClass.RealTime;
                    GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;
                }
            }
            catch { }
        }

        // --- ENGINE CLASS ---
        public class AdaptiveEngine : IDisposable
        {
            private InferenceSession Session;
            private OrtIoBinding IoBinding;
            private RunOptions RunOpts;

            // Memory Pins
            private GCHandle InputPin, OutputPin;
            private float[] OutputBuffer;

            // Dimensions
            private int _time, _channels, _batch;
            private int _strideFloats;

            // Stride Logic State
            private int _strideCounter;
            private int _currentK = 1;

            public AdaptiveEngine(string path, InferenceProvider provider, int batch, int channels, int time)
            {
                _time = time;
                _channels = channels;
                _batch = batch;
                _strideFloats = time * channels;

                // Assuming model outputs 1 float per batch item (Binary Classification)
                OutputBuffer = new float[batch];

                try
                {
                    // 1. Session Options (Optimized)
                    var opts = new SessionOptions();
                    if (provider == InferenceProvider.Cuda)
                    {
                        try { opts.AppendExecutionProvider_CUDA(0); } catch { Console.WriteLine("[ERR] CUDA Failed"); }
                    }
                    else if (provider == InferenceProvider.TensorRt)
                    {
                        try { opts.AppendExecutionProvider_Tensorrt(0); opts.AppendExecutionProvider_CUDA(0); } catch { }
                    }
                    else
                    {
                        opts.IntraOpNumThreads = 1;
                        opts.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    }
                    opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

                    // 2. Init Session
                    Session = new InferenceSession(path, opts);
                    RunOpts = new RunOptions();
                    IoBinding = Session.CreateIoBinding();

                    // 3. Pin Memory
                    var inputBuffer = new float[batch * _strideFloats];
                    InputPin = GCHandle.Alloc(inputBuffer, GCHandleType.Pinned);
                    OutputPin = GCHandle.Alloc(OutputBuffer, GCHandleType.Pinned);

                    // 4. Bind Tensor Wrappers
                    var mem = OrtMemoryInfo.DefaultInstance;

                    // Shape: [Batch, Time, Channels]
                    var inOrt = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(inputBuffer), new long[] { batch, time, channels });
                    var outOrt = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(OutputBuffer), new long[] { batch, 1 }); // [Batch, 1]

                    IoBinding.BindInput(Session.InputMetadata.Keys.First(), inOrt);
                    IoBinding.BindOutput(Session.OutputMetadata.Keys.First(), outOrt);

                    inOrt.Dispose(); outOrt.Dispose();
                }
                catch (Exception ex) { Console.WriteLine($"[FATAL] {ex.Message}"); throw; }
            }

            // --- STRIDE LOGIC ---
            public bool CheckStride()
            {
                if (_currentK <= 1) return true;
                _strideCounter++;
                if (_strideCounter >= _currentK)
                {
                    _strideCounter = 0;
                    return true;
                }
                return false;
            }

            // --- EXECUTION BATCH 1 ---
            public RippleOut ExecuteBatch1(Mat signal, bool bnoOk, RippleStateMachineMatBool fsm, int kBelow, int kAt)
            {
                unsafe
                {
                    // Zero-copy Transpose
                    TransposeToBuffer(signal, (float*)InputPin.AddrOfPinnedObject());
                }

                Session.RunWithBinding(RunOpts, IoBinding);

                // Read Result directly from pinned buffer
                float sigP = OutputBuffer[0];
                float artP = 0f; // No artifact input

                // Update Logic
                var rippleOut = fsm.Update(sigP, artP, bnoOk, signal);

                // Update Stride for next frame
                _currentK = (rippleOut.State != RippleState.NoRipple) ? kAt : kBelow;
                rippleOut.StrideUsed = _currentK;

                return rippleOut;
            }

            // --- EXECUTION BATCH 2 ---
            public RippleOut ExecuteBatch2(Mat signal, Mat artifact, bool bnoOk, RippleStateMachineMatBool fsm, int kBelow, int kAt)
            {
                unsafe
                {
                    float* ptr = (float*)InputPin.AddrOfPinnedObject();
                    TransposeToBuffer(signal, ptr);
                    TransposeToBuffer(artifact, ptr + _strideFloats); // Offset by 1 batch item
                }

                Session.RunWithBinding(RunOpts, IoBinding);

                float sigP = OutputBuffer[0];
                float artP = OutputBuffer[1];

                var rippleOut = fsm.Update(sigP, artP, bnoOk, signal);

                // Update Stride for next frame
                _currentK = (rippleOut.State != RippleState.NoRipple) ? kAt : kBelow;
                rippleOut.StrideUsed = _currentK;

                return rippleOut;
            }

            // Optimized Transpose
            private unsafe void TransposeToBuffer(Mat src, float* dstPtr)
            {
                float* srcPtr = (float*)src.Data.ToPointer();
                int srcStep = src.Step / sizeof(float);

                int tMax = _time;
                int cMax = _channels;

                int dstIdx = 0;
                for (int t = 0; t < tMax; t++)
                {
                    for (int c = 0; c < cMax; c++)
                    {
                        dstPtr[dstIdx++] = srcPtr[(c * srcStep) + t];
                    }
                }
            }

            public void Dispose()
            {
                Session?.Dispose(); IoBinding?.Dispose(); RunOpts?.Dispose();
                if (InputPin.IsAllocated) InputPin.Free();
                if (OutputPin.IsAllocated) OutputPin.Free();
            }
        }
    }
}