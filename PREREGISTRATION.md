# PREREGISTRATION — QNN-P1 Phase 5 (ibm_fez hardware evaluation)

Committed prior to any Tier 2 execution. Config SHA256 fingerprints for all
referenced runs live in the committed config_*.json files in this repository.
Date locked: 2026-07-18. Deviations, if any, will be logged in the section
at the end of this file BEFORE unblinding results.

## Locked simulation findings (Tier 0 / Tier 1, complete before this commit)

F1. Chain-region effect: +5.0 pts (A, seed 8281, worst-day snapshot;
    generic-region qubits 0-7 = 0.7350 vs drift-selected chain
    125-140 = 0.7850). Controlled single-variable comparison.
F2. Chain-mapped noise neutrality: paired Tier0->Tier1 deltas straddle zero
    (A s8281: 0.0; A s2002: +4.0; C s8281: -4.0). On the selected chain,
    calibration noise is performance-neutral within seed variance. The
    uniform-tax hypothesis is rejected.
F3. Drift null: 89% calibration spread over the chain region produced a
    1.5-pt inverted difference (worst 0.785 vs best 0.770). Calibration
    timing does not measurably affect performance on the selected chain.
F4. Regression-to-mean is present; single-seed deltas are trajectory noise.
    Only paired distributions are interpreted.
F5. Tier 0 baseline: three-way tie (A 0.7390 +/- 0.0281, B 0.7385 +/- 0.0525,
    C 0.7445 +/- 0.0367; n=10 each).

## Tier 2 protocol

Evaluation-only. No on-device training or fine-tuning. Checkpoints:
A s8281 (worst-day-trained) and C s8281 (worst-day-trained), each evaluated
on the full 200-sample test set at 4096 shots via EstimatorV2,
resilience_level=1, dynamical decoupling on, chain confirmed by fresh
preflight (chains + transpile, zero SWAPs) on run day. Calibration
timestamp recorded at run time. Classical arm B is not run on hardware;
its comparison value is its Tier 0 result.

## Preregistered endpoints

P1 (primary): A_hw vs B (0.7385). Success criterion for the chain thesis:
    A_hw within [0.72, 0.85] (the sim seed spread around A's Tier 1
    values). A_hw materially below 0.72 indicates real-device effects
    (coherent errors, crosstalk, non-Markovian noise) exceed the
    depolarizing+relaxation model.
P2 (secondary): A_hw vs C_hw. Sim predicts no separation beyond seed
    variance (|A_hw - C_hw| <= 0.05). A specific deficit of A vs C
    beyond 0.05 indicts CZ-gate physics undersold by the noise model.
P3 (tertiary): |A_hw - A_tier1_fullshot_reeval| = the sim-to-hardware gap;
    reported as the headline calibration-archive validation number, with
    post-hoc Tier 1 re-simulation against the run-day snapshot.

## Analysis plan

Full-shot (4096) Tier 1 reevals of all compared checkpoints are computed
BEFORE the hardware run and committed. Hardware accuracies compared against
those. No endpoint thresholds altered after data collection. Null results
on any endpoint are reported as findings. All shot counts, calibration
stamps, and job IDs archived in-repo.

## Known limitations (declared now)

Single task family (qlabels); n<=2 seeds per Tier 1 cell; Open-plan
queuing means A and C jobs may see slightly different calibration states;
noise model omits crosstalk and non-Markovian effects by construction.

## Deviations log

(none yet)

## Deviations log (updated 2026-07-18, pre-full-run)

D1. Runtime options (resilience_level=1, dynamical decoupling) could not be
    applied via the pennylane-qiskit plugin internals (UserWarning at device
    construction). Full run proceeds with plugin-default mitigation, exact
    state unverified. Logged prior to full-run execution; probe (4 samples,
    hw_probe.json) executed under identical conditions and is retained.
D2. Council pre-mortem skipped in favor of same-day execution; probe-first
    protocol substituted as the mechanical-risk gate.

D4. 2026-07-18 hardware attempts consumed the full 10m cycle quota without
    completing: root cause identified as pennylane-qiskit submitting one
    Runtime job per circuit (~30s billed overhead each), making the
    plugin path infeasible on Open plan. Ten jobs from attempt 1 completed
    server-side (partial batch 0). Replaced with direct EstimatorV2
    single-job evaluation (eval_hardware_direct.py), dry-run verified to
    reproduce checkpoint accuracy exactly. Endpoints and anchors unchanged.
    Full run rescheduled to next quota cycle, meter-verified before submit.

## Amendment A1 (2026-07-19, BEFORE any Tier 2 execution, quota now 180m)

Extra usage approved (180 min/yr). Scope additions declared prior to
unblinding any hardware result:
E1. Primary A run replicated 3x total; report mean +/- std (device-noise
    error bar). Applies to the A/worst-day checkpoint only.
E2. Best-day A checkpoint also evaluated on hardware (tertiary drift
    endpoint, P3, now measured rather than sim-only).
Core endpoints P1/P2 and all anchors unchanged. Shots remain 4096
(reevals showed precision saturation; no increase). Reading of any
hardware result deferred until core A (x3) and C are complete.

## Amendment A2 (2026-07-19, pre-unblinding — cost correction)

Measured hardware cost = ~790 billed system-seconds (~13 min) per run,
far above the 1-3 min estimate. Revising E2: the best-day hardware run is
CANCELLED (simulation already covers the drift/tertiary endpoint; not
worth ~13 min). E1 (A x3 replication) retained. Final Tier 2 set:
A x3 + C x1 = ~52 billed min. Mitigation spec (resilience=1, DD) unchanged.

## Amendment A3 (2026-07-19, pre-unblinding — shot count)

Hardware shots reduced 4096 -> 1024, justified by measured precision
saturation: all four Tier 1 full-shot reevals differed from 1024-shot
training values by <= 0.5 pt (committed reeval JSONs). Sim anchors for
P3 comparison remain the 4096-shot reevals; the <= 0.5 pt saturation
margin is noted as measurement uncertainty. Sample count stays 200,
resilience=1 + DD unchanged. E1 replication cancelled (cost). Final
Tier 2 scope: A x1 + C x1, ~8-10 min each per shot arithmetic shown
in-session. Run A first; verify actual meter cost before submitting C.
