using System;
using System.Diagnostics;

namespace DeepBrainInterface.Core
{
    /// <summary>Allocation-free detection engine: decimation with phases, per-phase preprocessing chain and window ring,
    /// one or two ONNX runners (batched when the model allows, else one row at a time), a combiner, and a decision stage.
    /// Works on raw sample memory; the Bonsai shell only adapts the Mat. Counter phases match Bonsai.Dsp Buffer(1, Skip),
    /// Buffer(TimePoints, Stride) and ClockSync, so PhaseCount = 1 reproduces the operator chain sample for sample.</summary>
    public sealed unsafe class FrameProcessor : IDisposable
    {
        readonly ProcessorConfig cfg;
        readonly IDecisionStage decision;
        readonly int nCh, nPhase, nModels;
        readonly float[][] vec;                 // per phase: channel vector
        readonly IPreprocessStage[][] stages;   // per phase
        readonly WindowRing[] rings;            // per phase
        readonly OrtRunner[] runners;           // per model (null for a streaming model)
        readonly OrtStreamRunner[] streams;     // per model (null for a windowed model)
        readonly string[] infos;                // per model: "Windowed [1,44,8]" or "Streaming B=2, 14 states"
        readonly int ringLen;                   // TimePoints, or B when every model is streaming
        readonly bool anyStream;
        readonly bool[] batched;                // per model: true when the runner holds all phases in one batch
        readonly float[][] phaseProbs;          // per model: probability per phase
        readonly float[] modelProbs;            // per model: combined over phases (NaN when inactive)
        readonly float[] stageIn = new float[1];
        readonly float[] snapshot;
        long rawIndex;                          // raw columns seen since start
        int strideCounter;
        long decisionIndex;
        int activeModels; Combiner combiner; int liveVersion;
        bool lastTtl;
        ulong lastClock; long keptStep; int clockSeen;   // gap detection (streaming models only)
        int stateResets;
        static readonly double usPerTick = 1e6 / Stopwatch.Frequency;

        public FrameProcessor(ProcessorConfig config, IDecisionStage decisionStage)
        {
            cfg = config ?? throw new ArgumentNullException(nameof(config));
            decision = decisionStage ?? throw new ArgumentNullException(nameof(decisionStage));
            if (cfg.Channels == null || cfg.Channels.Length == 0) throw new ArgumentException("Channels must list at least one amplifier row", "Channels");
            if (cfg.DecimationFactor < 1) throw new ArgumentException("DecimationFactor must be >= 1", "DecimationFactor");
            if (cfg.PhaseCount < 1 || cfg.PhaseCount > cfg.DecimationFactor) throw new ArgumentException($"PhaseCount must be in 1..DecimationFactor ({cfg.DecimationFactor}), got {cfg.PhaseCount}", "PhaseCount");
            if (cfg.TimePoints < 1) throw new ArgumentException("TimePoints must be >= 1", "TimePoints");
            if (cfg.Stride < 1) throw new ArgumentException("Stride must be >= 1", "Stride");
            if (cfg.ModelPaths == null || cfg.ModelPaths.Length < 1 || cfg.ModelPaths.Length > 2) throw new ArgumentException("ModelPaths must hold one or two models", "ModelPaths");
            if (cfg.StageFactory == null) throw new ArgumentException("StageFactory is required (use p => new IPreprocessStage[0] for none)", "StageFactory");
            nCh = cfg.Channels.Length; nPhase = cfg.PhaseCount; nModels = cfg.ModelPaths.Length;
            SetActiveModels(cfg.ActiveModels); SetCombiner(cfg.Combiner); liveVersion = 0;

            var isStream = new bool[nModels]; int block = 0;
            for (int m = 0; m < nModels; m++)
            {
                isStream[m] = OrtStreamRunner.IsStreaming(cfg.ModelPaths[m]);
                if (isStream[m]) anyStream = true;
            }
            if (anyStream && nPhase != 1) throw new ArgumentException($"streaming models need PhaseCount 1, got {nPhase}", "PhaseCount");
            vec = new float[nPhase][]; stages = new IPreprocessStage[nPhase][]; rings = new WindowRing[nPhase];
            for (int p = 0; p < nPhase; p++)
            {
                vec[p] = new float[nCh]; stages[p] = cfg.StageFactory(p) ?? new IPreprocessStage[0]; 
                foreach (var s in stages[p]) if (s.Channels != nCh) throw new ArgumentException($"preprocessing stage {s.GetType().Name} has {s.Channels} channels, expected {nCh}", "StageFactory");
            }
            runners = new OrtRunner[nModels]; streams = new OrtStreamRunner[nModels]; infos = new string[nModels];
            batched = new bool[nModels]; phaseProbs = new float[nModels][]; modelProbs = new float[nModels];
            try
            {
                for (int m = 0; m < nModels; m++)
                {
                    phaseProbs[m] = new float[nPhase];
                    if (isStream[m])
                    {
                        streams[m] = new OrtStreamRunner(cfg.ModelPaths[m], nCh);
                        int b = streams[m].Block;
                        if (block != 0 && b != block) throw new ArgumentException($"both streaming models need the same block size, got B={block} and B={b}", "ModelPaths");
                        block = b;
                        if (cfg.Stride != b) throw new ArgumentException($"Stride must equal the streaming block size B={b} of {cfg.ModelPaths[m]}, got {cfg.Stride}", "Stride");
                        infos[m] = streams[m].Info;
                        continue;
                    }
                    var shape = OrtRunner.ReadInputShape(cfg.ModelPaths[m]);
                    batched[m] = nPhase == 1 || shape.batch == -1 || shape.batch == nPhase;
                    runners[m] = new OrtRunner(cfg.ModelPaths[m], batched[m] ? nPhase : 1, cfg.TimePoints, nCh);
                    infos[m] = $"Windowed [1,{cfg.TimePoints},{nCh}]";
                }
            }
            catch { Dispose(); throw; }
            bool allStream = true; for (int m = 0; m < nModels; m++) allStream &= isStream[m];
            ringLen = allStream ? block : cfg.TimePoints;
            for (int p = 0; p < nPhase; p++) rings[p] = new WindowRing(ringLen, nCh);
            snapshot = cfg.SnapshotEnabled ? new float[ringLen * nCh] : null;
            strideCounter = cfg.Stride - 1;   // first full window emits at once (Bonsai.Dsp Buffer semantics)
        }

        public long DecisionIndex { get { return decisionIndex; } }
        /// <summary>State resets of the streaming models since construction (clock gaps).</summary>
        public int StateResets { get { return stateResets; } }
        /// <summary>"Windowed [1,44,8]" or "Streaming B=2, 14 states" for model m.</summary>
        public string ModelInfo(int m) { return infos[m]; }
        public int ActiveModels { get { return activeModels; } }
        public Combiner Combiner { get { return combiner; } }

        public void SetActiveModels(int value)
        {
            if (value < 0 || value > 2) throw new ArgumentOutOfRangeException(nameof(value));
            if (value >= 1 && nModels < 2) throw new ArgumentException("ActiveModels 1 or 2 needs two ModelPaths", "ActiveModels");
            if (runners != null && value != activeModels) liveVersion++;
            activeModels = value;
        }

        public void SetCombiner(Combiner value) { if (runners != null && value != combiner) liveVersion++; combiner = value; }

        /// <summary>Processes one frame of unsigned 16-bit samples (rows = channels, cols = samples, row stride in bytes).
        /// Writes up to cols decisions into outBuf and returns how many.</summary>
        public int ProcessFrame(ushort* data, int rows, int cols, int stepBytes, ulong* clocks, bool gate, RippleOut[] outBuf)
        {
            CheckRows(rows);
            int n = 0;
            for (int t = 0; t < cols; t++)
            {
                long ri = rawIndex++;
                int phase = (int)(ri % cfg.DecimationFactor);
                if (phase >= nPhase) continue;
                float[] v = vec[phase];
                for (int c = 0; c < nCh; c++) v[c] = *(ushort*)((byte*)data + cfg.Channels[c] * stepBytes + t * sizeof(ushort));
                if (Step(phase, clocks[t], gate, outBuf, ref n)) { }
            }
            return n;
        }

        /// <summary>Same for float32 samples (row stride in bytes).</summary>
        public int ProcessFrameF32(float* data, int rows, int cols, int stepBytes, ulong* clocks, bool gate, RippleOut[] outBuf)
        {
            CheckRows(rows);
            int n = 0;
            for (int t = 0; t < cols; t++)
            {
                long ri = rawIndex++;
                int phase = (int)(ri % cfg.DecimationFactor);
                if (phase >= nPhase) continue;
                float[] v = vec[phase];
                for (int c = 0; c < nCh; c++) v[c] = *(float*)((byte*)data + cfg.Channels[c] * stepBytes + t * sizeof(float));
                Step(phase, clocks[t], gate, outBuf, ref n);
            }
            return n;
        }

        /// <summary>Same for signed 16-bit samples.</summary>
        public int ProcessFrameS16(short* data, int rows, int cols, int stepBytes, ulong* clocks, bool gate, RippleOut[] outBuf)
        {
            CheckRows(rows);
            int n = 0;
            for (int t = 0; t < cols; t++)
            {
                long ri = rawIndex++;
                int phase = (int)(ri % cfg.DecimationFactor);
                if (phase >= nPhase) continue;
                float[] v = vec[phase];
                for (int c = 0; c < nCh; c++) v[c] = *(short*)((byte*)data + cfg.Channels[c] * stepBytes + t * sizeof(short));
                Step(phase, clocks[t], gate, outBuf, ref n);
            }
            return n;
        }

        void CheckRows(int rows)
        {
            for (int c = 0; c < nCh; c++)
                if (cfg.Channels[c] < 0 || cfg.Channels[c] >= rows) throw new ArgumentException($"Channels[{c}] = {cfg.Channels[c]} is outside the frame's {rows} rows", "Channels");
        }

        /// <summary>One kept sample of `phase` is in vec[phase]: run the chain, push the ring, and on phase 0 drive the stride counter.</summary>
        bool Step(int phase, ulong clock, bool gate, RippleOut[] outBuf, ref int n)
        {
            float[] v = vec[phase];
            if (anyStream && phase == 0) CheckClock(clock);
            var st = stages[phase];
            for (int i = 0; i < st.Length; i++) st[i].Update(v);
            rings[phase].Push(v);
            if (phase != 0) return false;
            if (!rings[0].IsFull) return false;   // phase 0 sets the cadence; later phases join once their own ring is full
            if (++strideCounter < cfg.Stride) return false;
            strideCounter = 0;
            int am = activeModels; Combiner cb = combiner;       // live properties read once per inference
            long t0 = Stopwatch.GetTimestamp();
            Infer(am, cb);
            float inferUs = (float)((Stopwatch.GetTimestamp() - t0) * usPerTick);
            float signal = Combine(modelProbs, am, cb);
            stageIn[0] = signal;
            var r = decision.Update(stageIn, gate, clock);
            r.Prob0 = modelProbs[0]; r.Prob1 = nModels > 1 ? modelProbs[1] : float.NaN;
            r.DecisionIndex = decisionIndex++;
            r.ConfigVersion = decision.ConfigVersion + liveVersion;
            r.InferenceUs = inferUs; r.StateResets = stateResets;
            if (snapshot != null && r.TTL && !lastTtl) { rings[0].CopyWindow(snapshot, 0); r.SnapshotWindow = snapshot; }
            lastTtl = r.TTL;
            outBuf[n++] = r;
            return true;
        }

        /// <summary>Streaming models: the expected kept-sample clock step is the smallest positive step seen so far; a step above 1.5x that,
        /// or a clock that does not advance, is an acquisition gap: states back to their initial values, ring and stride restart.</summary>
        void CheckClock(ulong clock)
        {
            if (clockSeen > 0)
            {
                long step = (long)(clock - lastClock);
                if (clockSeen == 1 && step > 0) { keptStep = step; clockSeen = 2; }
                else if (step > 0 && step < keptStep) keptStep = step;   // a gap inside the first step must not inflate the rule
                else if (step <= 0 || (clockSeen == 2 && step * 2 > keptStep * 3))
                {
                    for (int m = 0; m < nModels; m++) streams[m]?.ResetStates();
                    for (int p = 0; p < nPhase; p++) rings[p].Reset();
                    strideCounter = cfg.Stride - 1;
                    stateResets++;
                }
            }
            else clockSeen = 1;
            lastClock = clock;
        }

        void Infer(int am, Combiner cb)
        {
            for (int m = 0; m < nModels; m++)
            {
                bool active = am == 2 || am == m;
                if (!active) { modelProbs[m] = float.NaN; continue; }
                var sr = streams[m];
                if (sr != null) { rings[0].CopyTail(sr.Input, 0, sr.Block); sr.Run(); modelProbs[m] = sr.Output[sr.Block - 1]; continue; }
                var run = runners[m];
                if (batched[m])
                {
                    for (int p = 0; p < nPhase; p++) rings[p].CopyWindow(run.Input, p * cfg.TimePoints * nCh);
                    run.Run();
                    for (int p = 0; p < nPhase; p++) phaseProbs[m][p] = rings[p].IsFull ? run.Output[p] : float.NaN;
                }
                else
                {
                    for (int p = 0; p < nPhase; p++) { if (!rings[p].IsFull) { phaseProbs[m][p] = float.NaN; continue; } rings[p].CopyWindow(run.Input, 0); run.Run(); phaseProbs[m][p] = run.Output[0]; }
                }
                modelProbs[m] = CombineArr(phaseProbs[m], nPhase, cb);
            }
        }

        /// <summary>Combines the per-phase probabilities, ignoring NaN (phases whose ring is not yet full). Phase 0 is always present.</summary>
        float CombineArr(float[] a, int count, Combiner cb)
        {
            switch (cb)
            {
                case Combiner.Max: { float mx = a[0]; for (int i = 1; i < count; i++) if (!float.IsNaN(a[i]) && a[i] > mx) mx = a[i]; return mx; }
                case Combiner.Mean: { double s = 0; int k = 0; for (int i = 0; i < count; i++) if (!float.IsNaN(a[i])) { s += a[i]; k++; } return (float)(s / k); }
                default: return a[0];
            }
        }

        float Combine(float[] probs, int active, Combiner cb)
        {
            if (active == 0) return probs[0];
            if (active == 1) return probs[1];
            switch (cb)
            {
                case Combiner.Max: return Math.Max(probs[0], probs[1]);
                case Combiner.Mean: return 0.5f * (probs[0] + probs[1]);
                default: return probs[0];
            }
        }

        public void Dispose()
        {
            if (runners != null) foreach (var r in runners) r?.Dispose();
            if (streams != null) foreach (var r in streams) r?.Dispose();
        }
    }
}
