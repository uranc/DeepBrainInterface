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
    [Combinator]
    [Description("Stride→ONNX→FSM in one node. Accepts BNO as bool/int/byte/float/double/Mat. Optional post-event block skips only this node’s emissions.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class AdaptiveRippleDetector
    {
        // ===== Model =====
        [Category("Model")] public string ModelPath { get; set; } = @"C:\path\to\ripple_detector.onnx";
        [Category("Model")] public int NumThreads { get; set; } = 5;
        [Category("Model")] public int TimePoints { get; set; } = 44;
        [Category("Model")] public int Channels { get; set; } = 8;

        // ===== Thresholds =====
        [Category("Thresholds"), DisplayName("1. Gate (arm)")] public float GateThreshold { get; set; } = 0.10f;
        [Category("Thresholds"), DisplayName("2. Enter (+0.5)")] public float EnterThreshold { get; set; } = 0.50f;
        [Category("Thresholds"), DisplayName("3. Confirm (+1)")] public float ConfirmThreshold { get; set; } = 0.80f;
        [Category("Thresholds"), DisplayName("4. Event Score (≥)")]
        public float EventScoreThreshold { get; set; } = 2.5f;

        // ===== General =====
        [Category("General"), DisplayName("Detection Enabled")] public bool DetectionEnabled { get; set; } = true;

        // ===== Stride policy (internal K) =====
        [Category("Stride K"), DisplayName("K below Gate (< Gate)")] public int KBelowGate { get; set; } = 4;
        [Category("Stride K"), DisplayName("K at Gate (Gate ≤ d < Enter)")] public int KAtGate { get; set; } = 2;
        [Category("Stride K"), DisplayName("K at Enter (d ≥ Enter)")] public int KAtEnter { get; set; } = 1;
        [Category("Stride K"), DisplayName("Force Gate K After Ripple")] public bool ForceGateKAfterRipple { get; set; } = true;

        // ===== Optional post-event block =====
        [Category("Refractory"), DisplayName("Post-Event Block (ms)")]
        [Description("Skips this node's emissions for this duration after the event; 0 disables.")]
        public int PostEventBlockMs { get; set; } = 0;
        [Category("Refractory"), DisplayName("Block On Detection")]
        public bool BlockOnDetection { get; set; } = true;

        // ===== BNO numeric→bool threshold (for non-bool inputs) =====
        [Category("BNO"), DisplayName("BNO Numeric True Threshold")]
        [Description("For numeric/Mat BNO inputs, consider true if value ≥ this threshold.")]
        public float BnoTrueThreshold { get; set; } = 0.5f;

        // ===== Internal engine =====
        InferenceSession _session; string _inputName;
        float[] _buf; GCHandle _pin;
        readonly List<NamedOnnxValue> _inputs = new List<NamedOnnxValue>(1);

        int _kCurrent, _strideCounter;
        RippleState _state = RippleState.NoRipple, _prevState = RippleState.NoRipple;
        int _scoreTicks, _eventCount; float _lastEventScore;

        readonly Stopwatch _clock = new Stopwatch();
        long _blockUntilMs;

        void EnsureSession()
        {
            if (_session != null) return;
            _kCurrent = KBelowGate; _strideCounter = 0; _blockUntilMs = 0;

            _buf = new float[TimePoints * Channels];
            _pin = GCHandle.Alloc(_buf, GCHandleType.Pinned);

            var opts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = NumThreads,
                InterOpNumThreads = 1,
                EnableCpuMemArena = true
            };
            opts.AddSessionConfigEntry("session.enable_mem_pattern", "1");
            opts.AddSessionConfigEntry("session.execution_mode", "ORT_SEQUENTIAL");
            opts.AddSessionConfigEntry("cpu.arena_extend_strategy", "kSameAsRequested");
            opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

            _session = new InferenceSession(ModelPath, opts);
            _inputName = _session.InputMetadata.Keys.First();

            var warm = new DenseTensor<float>(_buf, new[] { 1, TimePoints, Channels });
            using (var _ = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, warm) })) { }

            _clock.Restart();
        }

        DenseTensor<float> Pack(Mat m)
        {
            int t = m.Rows, c = m.Cols;
            if (!((t == TimePoints && c == Channels) || (t == Channels && c == TimePoints)))
                throw new ArgumentException($"Mat shape {m.Rows}×{m.Cols} != {TimePoints}×{Channels}");
            unsafe
            {
                var dst = (float*)_pin.AddrOfPinnedObject().ToPointer();
                var src = (float*)m.Data.ToPointer();
                if (m.Rows == TimePoints && m.Cols == Channels)
                    Buffer.MemoryCopy(src, dst, _buf.Length * sizeof(float), _buf.Length * sizeof(float));
                else
                    for (int cc = 0; cc < Channels; cc++)
                        for (int tt = 0; tt < TimePoints; tt++)
                            dst[tt * Channels + cc] = src[cc * TimePoints + tt];
            }
            return new DenseTensor<float>(_buf, new[] { 1, TimePoints, Channels });
        }

        float Infer(Mat m)
        {
            var tensor = Pack(m);
            var named = NamedOnnxValue.CreateFromTensor(_inputName, tensor);
            _inputs.Clear(); _inputs.Add(named);
            using (var r = _session.Run(_inputs))
            {
                var outT = r.First().AsTensor<float>();
                return outT.ToArray()[0];
            }
        }

        // ===== Public overloads (Mat only, assume BNO OK) =====
        public IObservable<RippleOut> Process(IObservable<Mat> source)
        {
            EnsureSession();
            return source.Where(StrideGate).Select(m => Tick(Infer(m), true));
        }

        // ===== Public overloads (Mat + BNO as various types) =====
        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, bool>> source)
        {
            EnsureSession();
            return source.Where(t => StrideGate(t.Item1))
                         .Select(t => Tick(Infer(t.Item1), t.Item2));
        }

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, int>> source)
        {
            EnsureSession();
            return source.Where(t => StrideGate(t.Item1))
                         .Select(t => Tick(Infer(t.Item1), t.Item2 != 0));
        }

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, byte>> source)
        {
            EnsureSession();
            return source.Where(t => StrideGate(t.Item1))
                         .Select(t => Tick(Infer(t.Item1), t.Item2 != 0));
        }

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, float>> source)
        {
            EnsureSession();
            float thr = BnoTrueThreshold;
            return source.Where(t => StrideGate(t.Item1))
                         .Select(t => Tick(Infer(t.Item1), t.Item2 >= thr));
        }

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, double>> source)
        {
            EnsureSession();
            double thr = BnoTrueThreshold;
            return source.Where(t => StrideGate(t.Item1))
                         .Select(t => Tick(Infer(t.Item1), t.Item2 >= thr));
        }

        public IObservable<RippleOut> Process(IObservable<Tuple<Mat, Mat>> source)
        {
            EnsureSession();
            float thr = BnoTrueThreshold;
            return source.Where(t => StrideGate(t.Item1))
                         .Select(t => Tick(Infer(t.Item1), ReadScalar(t.Item2) >= thr));
        }

        // (Optional) support ValueTuple from some operators
        public IObservable<RippleOut> Process(IObservable<(Mat, bool)> source)
        {
            EnsureSession();
            return source.Where(t => StrideGate(t.Item1))
                         .Select(t => Tick(Infer(t.Item1), t.Item2));
        }
        public IObservable<RippleOut> Process(IObservable<(Mat, int)> source)
        {
            EnsureSession();
            return source.Where(t => StrideGate(t.Item1))
                         .Select(t => Tick(Infer(t.Item1), t.Item2 != 0));
        }
        public IObservable<RippleOut> Process(IObservable<(Mat, float)> source)
        {
            EnsureSession();
            float thr = BnoTrueThreshold;
            return source.Where(t => StrideGate(t.Item1))
                         .Select(t => Tick(Infer(t.Item1), t.Item2 >= thr));
        }

        // ===== Gate (stride + optional post-event block) =====
        bool StrideGate(Mat _)
        {
            if (PostEventBlockMs > 0 && _clock.ElapsedMilliseconds < _blockUntilMs)
                return false;

            int k = _kCurrent;
            if (k <= 0) return false;
            if (k == 1) return true;

            _strideCounter++;
            if (_strideCounter >= k) { _strideCounter = 0; return true; }
            return false;
        }

        // ===== FSM + K policy =====
        RippleOut Tick(float prob, bool bnoOk)
        {
            float gate = Clamp01(GateThreshold), enter = Clamp01(EnterThreshold), confirm = Clamp01(ConfirmThreshold);
            int eventTicks = (int)System.Math.Ceiling(EventScoreThreshold * 2f);

            bool gatesOn = DetectionEnabled && bnoOk;
            float d = prob;
            bool eventPulse = false;

            _prevState = _state;

            if (!gatesOn) { _state = RippleState.NoRipple; _scoreTicks = 0; }
            else
            {
                switch (_state)
                {
                    case RippleState.NoRipple:
                        if (d >= gate) { _state = RippleState.Possible; _scoreTicks = 0; }
                        break;

                    case RippleState.Possible:
                        if (d < gate) { _state = RippleState.NoRipple; _scoreTicks = 0; }
                        else
                        {
                            if (d >= confirm) _scoreTicks += 2;   // +1.0
                            else if (d >= enter) _scoreTicks += 1;   // +0.5
                            else _scoreTicks = 0;

                            if (_scoreTicks >= eventTicks)
                            {
                                _state = RippleState.Ripple;
                                _eventCount++;
                                _lastEventScore = _scoreTicks * 0.5f;
                                _scoreTicks = 0;
                                eventPulse = true;

                                if (PostEventBlockMs > 0 && BlockOnDetection)
                                    _blockUntilMs = _clock.ElapsedMilliseconds + PostEventBlockMs;
                            }
                        }
                        break;

                    case RippleState.Ripple:
                        if (d >= confirm) { /* stay */ }
                        else if (d >= gate) { _state = RippleState.Possible; _scoreTicks = 0; }
                        else { _state = RippleState.NoRipple; _scoreTicks = 0; }
                        break;
                }
            }

            if (PostEventBlockMs > 0 && !BlockOnDetection &&
                _prevState == RippleState.Ripple && _state != RippleState.Ripple)
            {
                _blockUntilMs = _clock.ElapsedMilliseconds + PostEventBlockMs;
            }

            int kNew;
            if (!gatesOn || d < gate) kNew = KBelowGate;
            else if (_prevState == RippleState.Ripple && _state != RippleState.Ripple && ForceGateKAfterRipple)
                kNew = KAtGate;
            else if (d < enter) kNew = KAtGate;
            else kNew = KAtEnter;

            if (kNew < 0) kNew = 0;
            _kCurrent = kNew;

            return new RippleOut
            {
                State = _state,
                Score = _scoreTicks * 0.5f,
                DecisionValue = d,
                Skip = _kCurrent,
                EventCount = _eventCount,
                LastEventScore = _lastEventScore,
                EventPulse = eventPulse,
                TTL = (_state == RippleState.Ripple)
            };
        }

        static float Clamp01(float v) { if (v < 0f) return 0f; if (v > 1f) return 1f; return v; }

        static float ReadScalar(Mat m)
        {
            if (m == null) return 0f;
            unsafe
            {
                switch (m.Depth)
                {
                    case Depth.F32: return *((float*)m.Data.ToPointer());
                    case Depth.U8: return *((byte*)m.Data.ToPointer());
                    case Depth.S32: return *((int*)m.Data.ToPointer());
                    case Depth.F64: return (float)(*((double*)m.Data.ToPointer()));
                    default: return 0f;
                }
            }
        }
    }
}
