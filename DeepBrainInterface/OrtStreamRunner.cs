using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace DeepBrainInterface.Core
{
    /// <summary>Streaming ONNX model (pytorch-tcn style): data input [1, B, C] float32, probability output [1, B] or [1, B, 1],
    /// and state tensors: every input X with an output X_out (same shape, float32 or float64). Initial state values come from the
    /// model metadata key "state_init" ("name=value,name=value"); unlisted states start at 0, unknown names are an error.
    /// Two IoBindings are built once (A: states set0 in -> set1 out, B: set1 in -> set0 out) and alternated, so each call reads the
    /// states the previous call wrote with no copies and no allocations. Fill <see cref="Input"/> (sample-major), call <see cref="Run"/>,
    /// read <see cref="Output"/>[B-1] for the newest sample.</summary>
    public sealed class OrtStreamRunner : IDisposable
    {
        public const string StateInitKey = "state_init";
        public const string StateSuffix = "_out";

        readonly InferenceSession session;
        readonly RunOptions runOpts;
        readonly OrtIoBinding[] bindings = new OrtIoBinding[2];
        readonly List<GCHandle> handles = new List<GCHandle>();
        readonly List<OrtValue> values = new List<OrtValue>();
        readonly Array[][] states;      // [state][set] float[] or double[]
        readonly double[] initValues;
        int cur;                         // binding (and state set) the next Run reads

        public float[] Input { get; }
        public float[] Output { get; }
        public int Block { get; }
        public int Channels { get; }
        public int StateCount { get { return states.Length; } }
        public string Info { get { return "Streaming B=" + Block + ", " + StateCount + " states"; } }

        /// <summary>True when the model has at least one state pair (input X with output X_out).</summary>
        public static bool IsStreaming(string modelPath)
        {
            using (var s = new InferenceSession(modelPath))
            {
                var outs = new HashSet<string>(s.OutputMetadata.Keys);
                return s.InputMetadata.Keys.Any(k => outs.Contains(k + StateSuffix));
            }
        }

        public OrtStreamRunner(string modelPath, int channels)
        {
            Channels = channels;
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
            try { session = new InferenceSession(modelPath, opts); } finally { opts.Dispose(); }
            try
            {
                var ins = session.InputMetadata; var outs = session.OutputMetadata;
                var stateNames = ins.Keys.Where(k => outs.ContainsKey(k + StateSuffix)).ToArray();
                var dataIns = ins.Keys.Except(stateNames).ToArray();
                var probOuts = outs.Keys.Except(stateNames.Select(n => n + StateSuffix)).ToArray();
                if (stateNames.Length == 0) throw new ArgumentException("not a streaming model: no input X with an output X" + StateSuffix);
                if (dataIns.Length != 1 || probOuts.Length != 1)
                    throw new ArgumentException("streaming model must have exactly one data input and one probability output besides the state pairs; " +
                        "unpaired inputs: " + string.Join(", ", dataIns) + "; unpaired outputs: " + string.Join(", ", probOuts));
                string dataName = dataIns[0], probName = probOuts[0];
                var d = ins[dataName];
                if (d.ElementType != typeof(float) || d.Dimensions.Length != 3 || d.Dimensions[0] != 1 || d.Dimensions[1] < 1)
                    throw new ArgumentException("data input " + dataName + " must be float32 [1, B, C], got [" + string.Join(",", d.Dimensions) + "]");
                Block = d.Dimensions[1];
                if (d.Dimensions[2] != channels) throw new ArgumentException("data input " + dataName + " has " + d.Dimensions[2] + " channels, the node selects " + channels);
                var od = outs[probName].Dimensions;
                long nOut = 1; foreach (var x in od) nOut *= x > 0 ? x : 1;
                if (outs[probName].ElementType != typeof(float) || od[0] != 1 || nOut != Block)
                    throw new ArgumentException("probability output " + probName + " must be float32 [1, " + Block + "] or [1, " + Block + ", 1], got [" + string.Join(",", od) + "]");

                // initial values
                var init = new Dictionary<string, double>();
                string spec;
                if (session.ModelMetadata.CustomMetadataMap.TryGetValue(StateInitKey, out spec) && !string.IsNullOrWhiteSpace(spec))
                    foreach (var kv in spec.Split(','))
                    {
                        var parts = kv.Split('=');
                        double v;
                        if (parts.Length != 2 || !double.TryParse(parts[1].Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out v))
                            throw new ArgumentException(StateInitKey + " entry '" + kv + "' is not name=value");
                        var name = parts[0].Trim();
                        if (!stateNames.Contains(name)) throw new ArgumentException(StateInitKey + " names unknown state '" + name + "'; states: " + string.Join(", ", stateNames));
                        init[name] = v;
                    }

                var mem = OrtMemoryInfo.DefaultInstance;
                Input = new float[Block * channels];
                Output = new float[Block];
                var vIn = Pin(mem, Input, new long[] { 1, Block, channels });
                var vOut = Pin(mem, Output, od.Select(x => (long)(x > 0 ? x : 1)).ToArray());
                states = new Array[stateNames.Length][];
                initValues = new double[stateNames.Length];
                var stVals = new OrtValue[stateNames.Length, 2];
                for (int i = 0; i < stateNames.Length; i++)
                {
                    var m = ins[stateNames[i]]; var mo = outs[stateNames[i] + StateSuffix];
                    if (m.ElementType != mo.ElementType || !m.Dimensions.SequenceEqual(mo.Dimensions))
                        throw new ArgumentException("state " + stateNames[i] + " and " + stateNames[i] + StateSuffix + " differ in type or shape");
                    if (m.Dimensions.Any(x => x < 1)) throw new ArgumentException("state " + stateNames[i] + " has a dynamic dimension");
                    var shape = m.Dimensions.Select(x => (long)x).ToArray();
                    int n = m.Dimensions.Aggregate(1, (a, b) => a * b);
                    states[i] = new Array[2];
                    initValues[i] = init.ContainsKey(stateNames[i]) ? init[stateNames[i]] : 0.0;
                    for (int s = 0; s < 2; s++)
                    {
                        if (m.ElementType == typeof(float)) { var a = new float[n]; states[i][s] = a; stVals[i, s] = Pin(mem, a, shape); }
                        else if (m.ElementType == typeof(double)) { var a = new double[n]; states[i][s] = a; stVals[i, s] = Pin(mem, a, shape); }
                        else throw new ArgumentException("state " + stateNames[i] + " must be float32 or float64, is " + m.ElementType.Name);
                    }
                }
                for (int b = 0; b < 2; b++)
                {
                    var bind = session.CreateIoBinding();
                    bind.BindInput(dataName, vIn); bind.BindOutput(probName, vOut);
                    for (int i = 0; i < stateNames.Length; i++)
                    {
                        bind.BindInput(stateNames[i], stVals[i, b]);
                        bind.BindOutput(stateNames[i] + StateSuffix, stVals[i, 1 - b]);
                    }
                    bindings[b] = bind;
                }
                runOpts = new RunOptions();
                ResetStates();
                for (int w = 0; w < 50; w++) Run();
                Array.Clear(Output, 0, Output.Length);
                ResetStates();
            }
            catch { Dispose(); throw; }
        }

        OrtValue Pin<T>(OrtMemoryInfo mem, T[] arr, long[] shape) where T : unmanaged
        {
            handles.Add(GCHandle.Alloc(arr, GCHandleType.Pinned));
            var v = OrtValue.CreateTensorValueFromMemory(mem, new Memory<T>(arr), shape);
            values.Add(v);
            return v;
        }

        /// <summary>Puts every state at its initial value in the set the next <see cref="Run"/> reads.</summary>
        public void ResetStates()
        {
            for (int i = 0; i < states.Length; i++)
            {
                var a = states[i][cur];
                var f = a as float[];
                if (f != null) { float v = (float)initValues[i]; for (int k = 0; k < f.Length; k++) f[k] = v; }
                else { var dd = (double[])a; double v = initValues[i]; for (int k = 0; k < dd.Length; k++) dd[k] = v; }
            }
        }

        public void Run() { session.RunWithBinding(runOpts, bindings[cur]); cur ^= 1; }

        public void Dispose()
        {
            foreach (var b in bindings) b?.Dispose();
            runOpts?.Dispose();
            foreach (var v in values) v.Dispose();
            foreach (var h in handles) if (h.IsAllocated) h.Free();
            session?.Dispose();
        }
    }
}
