# DeepBrainInterface

Bonsai package for real-time closed-loop ripple detection on ONIX (OpenEphys.Onix1) with an ONNX TCN model and a hardware TTL
output. Version 1.1.79.

## Packages in this repository

- `DeepBrainInterface.Core` (netstandard2.0, no Bonsai/OpenCV dependency): the detection math. Preprocessing stages
  (`ZScoreStage`, `EmaStage`, `ReReferenceStage`, `CsdStage`, `HighpassStage`), `OrtRunner` (ONNX Runtime with bound pinned
  buffers), decision stages (`RippleFsm`, `ThresholdHoldStage`), `WindowRing`, and `FrameProcessor`, which runs decimation,
  phases, preprocessing, inference and the decision on raw sample memory with no allocation after start.
- `DeepBrainInterface` (net472 Bonsai package): the nodes. The old chain (`SlidingWindowZScore`, `RippleDetectorCPU`,
  `ClockSync`, `RippleStateMachineMatBool`) is kept and now runs on the Core classes; `RippleSuperNode` replaces the chain.
- `DeepBrainInterface.Core.Tests` (net8.0, xunit, runs on Linux): unit tests and the replay test that reproduces the probabilities
  Bonsai recorded online (session 04112026). `DeepBrainInterface.Tests` (net472): node tests, run on the rig.

## RippleSuperNode

Input: `Zip(MemberSelector AmplifierData, MemberSelector Clock)` of `Rhd2164Data`, optionally `WithLatestFrom(gate bool)`.
Output: `RippleOut` per inference (same struct as `RippleStateMachineMatBool`, plus `ConfigVersion`, `Prob0`, `Prob1`,
`DecisionIndex`, `SnapshotWindow`). Wire the `DigitalOutput`, CsvWriter and subjects exactly as after the old FSM.

Fixed at start: `Channels` (0-based amplifier rows in model order), `DecimationFactor` (1 = raw rate), `PhaseCount`
(= `DecimationFactor` to process all shifted copies; needs a dynamic-batch ONNX export or runs them one by one),
`Preprocessing` (None / ZScore / Ema) with `ZScoreWindow`, `ZScoreEps`, `EmaAlpha`, `ModelPath`, `ModelPath2`, `TimePoints`,
`Stride`, `Decision` (Fsm / ThresholdHold), `UseHardwareClock`, `SnapshotEnabled`. Changing these requires a workflow restart.

Live: all FSM parameters, `ActiveModels` (0, 1, 2 = both), `Combiner` (First / Max / Mean). Every live change increments
`ConfigVersion` in the output, so the CSV shows when it took effect.

The model's input shape is read from the file; a mismatch with `TimePoints`/`Channels`, a second output, or a channel index
outside the frame produces an error naming the field.

## RippleDisplayBlock

`Mat` in (all amplifier rows), `Mat` out: rows reordered by `ChannelMap` (the probe order, one entry per input row, comma-separated),
then `Channels` picks rows out of that order (indices into the mapped order, empty = all; the output has one row per entry, so
"add a cortical contact" or "move deeper" is a change of this list only), every `DisplayDecimation`-th sample, `BlockSamples`
columns per block, F32, from a ring of `RingDepth` pre-allocated Mats. Replaces `SelectChannels`-based display paths, which
create one finalizable MatHeader per channel per sample. Block rate = 30 kHz / DisplayDecimation / BlockSamples; size RingDepth for at least 300 ms at that rate (the workflow uses 60 / 1 / 256 = 500 blocks per second, 512 ms).

## Workflows

- `workflows/phase0_supernode.bonsai`: the phase0 socket workflow with the chain replaced by `RippleSuperNode`, the display fed by
  `RippleDisplayBlock` (all 64 channels, probe order, µV), the decisions CSV extended with
  `ConfigVersion,Prob0,Prob1,DecisionIndex`, and `ripple_config_N.csv` written once per parameter change (DistinctUntilChangedBy
  on `ConfigVersion`) with the hardware `Clock` of the first decision under the new parameters, so every row of the decisions
  CSV joins to the parameters in force.

### 1.1.79 (property level only, data path unchanged; replay test bit-identical to 1.1.78)

- Channel lists edit as comma-separated text (`Channels`, `ChannelMap`).
- `PhaseCount` documented: P phases = P shifted decimated copies, each with its own preprocessing state and window, run as one
  batch [P, TimePoints, Channels] when the ONNX batch dimension allows it, else as P calls (the deployed models are batch 1).
- `ModelPath2` / `ActiveModels` marked optional (one model, `ActiveModels` 0, is the deployed setup); `Ema` marked experimental.
- Decision options split into sub-groups; FSM parameters back in their 1.1.76 groups (Thresholds, Scoring, Leaky Bucket, TTL Output).
- `DecayWaitMs`: with `UseHardwareClock`, the leaky-bucket grace can be given in ms of the headstage clock instead of frames.

## Why

An ETW trace (2026-11-05) showed the residual 5–8 ms trigger-latency band to be the CLR allocator's 5 ms back-off sleep when
realtime threads contend for the GC allocation lock; the Bonsai.Dsp operators created ~18,500 finalizable `MatHeader` objects
per second on the processing thread. The super node removes that allocation. Details in the SWRProject docs
(`docs/superpowers/specs/2026-11-05-ripple-supernode-design.md`, `docs/ideas_tracker.md` DEP-83).

## Build and test

```
dotnet test DeepBrainInterface.Core.Tests          # Linux or Windows; replay test needs DBI_REPLAY_DIR (see ReplayTests.cs)
dotnet build DeepBrainInterface -c Release        # produces the .nupkg
dotnet test DeepBrainInterface.Tests               # Windows (OpenCV.Net / ONNX Runtime natives)
```

## Behaviour changes versus 1.1.76 in the kept nodes

- `RippleDetectorCPU` now refuses models whose output is not one value per batch row, and always lets the ORT intra-op pool spin
  (`AllowSpinning` was already a documented no-op with one thread).
- `RippleStateMachineMatBool.TriggerDelayMs` behaves as before (the logged delay refreshes to the minimum immediately).
- Live FSM edits only reach the output when `Decision = Fsm`; with `ThresholdHold`, `PostRippleMs` is fixed at start.
