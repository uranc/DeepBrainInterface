# DeepBrainInterface

Bonsai package for real-time closed-loop ripple detection on ONIX (OpenEphys.Onix1) with an ONNX TCN model and a hardware TTL
output. Version 1.1.78.

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

`Mat` in (all amplifier rows), `Mat` out: rows reordered by `DisplayOrder`, every `DisplayDecimation`-th sample, `BlockSamples`
columns per block, F32, from a ring of `RingDepth` pre-allocated Mats. Replaces `SelectChannels`-based display paths, which
create one finalizable MatHeader per channel per sample.

## Workflows

- `workflows/phase0_supernode.bonsai`: the phase0 socket workflow with the chain replaced by `RippleSuperNode`, the display fed by
  `RippleDisplayBlock` (all 64 channels, probe order, µV), and the decisions CSV extended with
  `ConfigVersion,Prob0,Prob1,DecisionIndex`.

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
