using System;

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
        readonly OrtRunner[] runners;           // per model
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

            vec = new float[nPhase][]; stages = new IPreprocessStage[nPhase][]; rings = new WindowRing[nPhase];
            for (int p = 0; p < nPhase; p++)
            {
                vec[p] = new float[nCh]; stages[p] = cfg.StageFactory(p) ?? new IPreprocessStage[0]; rings[p] = new WindowRing(cfg.TimePoints, nCh);
                foreach (var s in stages[p]) if (s.Channels != nCh) throw new ArgumentException($"preprocessing stage {s.GetType().Name} has {s.Channels} channels, expected {nCh}", "StageFactory");
            }
            runners = new OrtRunner[nModels]; batched = new bool[nModels]; phaseProbs = new float[nModels][]; modelProbs = new float[nModels];
            try
            {
                for (int m = 0; m < nModels; m++)
                {
                    var shape = OrtRunner.ReadInputShape(cfg.ModelPaths[m]);
                    batched[m] = nPhase == 1 || shape.batch == -1 || shape.batch == nPhase;
                    runners[m] = new OrtRunner(cfg.ModelPaths[m], batched[m] ? nPhase : 1, cfg.TimePoints, nCh);
                    phaseProbs[m] = new float[nPhase];
                }
            }
            catch { Dispose(); throw; }
            snapshot = cfg.SnapshotEnabled ? new float[cfg.TimePoints * nCh] : null;
            strideCounter = cfg.Stride - 1;   // first full window emits at once (Bonsai.Dsp Buffer semantics)
        }

        public long DecisionIndex { get { return decisionIndex; } }
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
            var st = stages[phase];
            for (int i = 0; i < st.Length; i++) st[i].Update(v);
            rings[phase].Push(v);
            if (phase != 0) return false;
            if (!rings[0].IsFull) return false;   // phase 0 sets the cadence; later phases join once their own ring is full
            if (++strideCounter < cfg.Stride) return false;
            strideCounter = 0;
            Infer();
            float signal = Combine(modelProbs, activeModels);
            stageIn[0] = signal;
            var r = decision.Update(stageIn, gate, clock);
            r.Prob0 = modelProbs[0]; r.Prob1 = nModels > 1 ? modelProbs[1] : float.NaN;
            r.DecisionIndex = decisionIndex++;
            r.ConfigVersion = decision.ConfigVersion + liveVersion;
            if (snapshot != null && r.TTL && !lastTtl) { rings[0].CopyWindow(snapshot, 0); r.SnapshotWindow = snapshot; }
            lastTtl = r.TTL;
            outBuf[n++] = r;
            return true;
        }

        void Infer()
        {
            for (int m = 0; m < nModels; m++)
            {
                bool active = activeModels == 2 || activeModels == m;
                if (!active) { modelProbs[m] = float.NaN; continue; }
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
                modelProbs[m] = CombineArr(phaseProbs[m], nPhase);
            }
        }

        /// <summary>Combines the per-phase probabilities, ignoring NaN (phases whose ring is not yet full). Phase 0 is always present.</summary>
        float CombineArr(float[] a, int count)
        {
            switch (combiner)
            {
                case Combiner.Max: { float mx = a[0]; for (int i = 1; i < count; i++) if (!float.IsNaN(a[i]) && a[i] > mx) mx = a[i]; return mx; }
                case Combiner.Mean: { double s = 0; int k = 0; for (int i = 0; i < count; i++) if (!float.IsNaN(a[i])) { s += a[i]; k++; } return (float)(s / k); }
                default: return a[0];
            }
        }

        float Combine(float[] probs, int active)
        {
            if (active == 0) return probs[0];
            if (active == 1) return probs[1];
            switch (combiner)
            {
                case Combiner.Max: return Math.Max(probs[0], probs[1]);
                case Combiner.Mean: return 0.5f * (probs[0] + probs[1]);
                default: return probs[0];
            }
        }

        public void Dispose() { if (runners != null) foreach (var r in runners) r?.Dispose(); }
    }
}
