using System;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace DeepBrainInterface.Core
{
    /// <summary>ONNX Runtime session with pinned, pre-bound input [batch, timePoints, channels] and output [batch, 1].
    /// Lifted from RippleDetectorCPU.InitializeSession: 1 intra-op thread, sequential, denormals as zero, arena on, 50 warm-ups.
    /// Fill <see cref="Input"/> (sample-major: index = (b*timePoints + t)*channels + c), call <see cref="Run"/>, read <see cref="Output"/>[b].</summary>
    public sealed class OrtRunner : IDisposable
    {
        readonly InferenceSession session;
        readonly OrtIoBinding binding;
        readonly RunOptions runOpts;
        readonly GCHandle hIn, hOut;
        readonly OrtValue valIn, valOut;

        public float[] Input { get; }
        public float[] Output { get; }
        public int Batch { get; }
        public int TimePoints { get; }
        public int Channels { get; }

        /// <summary>Input dims of the model's first input; batch is -1 when symbolic (dynamic), else the fixed value.</summary>
        public static (int batch, int timePoints, int channels) ReadInputShape(string modelPath)
        {
            using (var s = new InferenceSession(modelPath))
            {
                var dims = s.InputMetadata.Values.First().Dimensions;
                if (dims.Length != 3) throw new ArgumentException("model input must be rank 3 [batch, timePoints, channels], got rank " + dims.Length);
                return (dims[0] > 0 ? dims[0] : -1, dims[1], dims[2]);
            }
        }

        /// <summary>The model's own batch dimension (-1 = dynamic). A fixed batch of 1 means callers must loop for multi-row batches.</summary>
        public int ModelBatch { get; private set; }

        public OrtRunner(string modelPath, int batch, int timePoints, int channels)
        {
            if (batch < 1) throw new ArgumentOutOfRangeException(nameof(batch));
            Batch = batch; TimePoints = timePoints; Channels = channels;
            var opts = new SessionOptions
            {
                IntraOpNumThreads = 1, InterOpNumThreads = 1,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                EnableCpuMemArena = true, EnableMemoryPattern = true
            };
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            opts.AddSessionConfigEntry("session.execution_mode", "ORT_SEQUENTIAL");
            opts.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
            opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");
            session = new InferenceSession(modelPath, opts);
            opts.Dispose();
            try
            {
                if (session.OutputMetadata.Count != 1)
                    throw new ArgumentException("model must have exactly one output; found " + string.Join(", ", session.OutputMetadata.Keys));
                if (session.InputMetadata.Count != 1) throw new ArgumentException("model must have exactly one input, has " + session.InputMetadata.Count);
                var inMeta = session.InputMetadata.Values.First();
                var d = inMeta.Dimensions;
                if (d.Length != 3) throw new ArgumentException("model input must be rank 3 [batch, timePoints, channels]");
                ModelBatch = d[0] > 0 ? d[0] : -1;
                if (d[0] > 0 && d[0] != batch) throw new ArgumentException($"model batch is fixed at {d[0]}, requested {batch}; export the model with a dynamic batch or run rows sequentially");
                if (d[1] != timePoints) throw new ArgumentException($"model timePoints is {d[1]}, requested {timePoints}");
                if (d[2] != channels) throw new ArgumentException($"model channels is {d[2]}, requested {channels}");
                var outMeta = session.OutputMetadata.Values.First();
                var od = outMeta.Dimensions;
                long outPer = 1; for (int i = 1; i < od.Length; i++) outPer *= od[i] > 0 ? od[i] : 1;
                if (outPer != 1) throw new ArgumentException("model output must be [batch, 1] (one probability per row); got [" + string.Join(",", od) + "] for output " + session.OutputMetadata.Keys.First());

                Input = new float[batch * timePoints * channels];
                Output = new float[batch];
                hIn = GCHandle.Alloc(Input, GCHandleType.Pinned);
                hOut = GCHandle.Alloc(Output, GCHandleType.Pinned);
                var mem = OrtMemoryInfo.DefaultInstance;
                var outShape = new long[od.Length]; for (int i = 0; i < od.Length; i++) outShape[i] = od[i] > 0 ? od[i] : (i == 0 ? batch : 1);
                valIn = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(Input), new long[] { batch, timePoints, channels });
                valOut = OrtValue.CreateTensorValueFromMemory(mem, new Memory<float>(Output), outShape);
                binding = session.CreateIoBinding();
                binding.BindInput(session.InputMetadata.Keys.First(), valIn);
                binding.BindOutput(session.OutputMetadata.Keys.First(), valOut);
                runOpts = new RunOptions();
                for (int i = 0; i < 50; i++) session.RunWithBinding(runOpts, binding);
            }
            catch { Dispose(); throw; }
        }

        public void Run() { session.RunWithBinding(runOpts, binding); }

        public void Dispose()
        {
            binding?.Dispose(); runOpts?.Dispose(); valIn?.Dispose(); valOut?.Dispose();
            if (hIn.IsAllocated) hIn.Free(); if (hOut.IsAllocated) hOut.Free();
            session?.Dispose();
        }
    }
}
