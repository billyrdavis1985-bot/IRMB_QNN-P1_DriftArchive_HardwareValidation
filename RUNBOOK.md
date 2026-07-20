# RUNBOOK — QNN-P1 on ibm_fez

Every step before Phase 4 is free. Do not skip preflight; it exists to catch the
failure modes that would otherwise cost queue time and shots.

---

## Phase 0 — Environment (once, ~15 min)

```bash
cd fez-qnn
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.lock          # pin BEFORE any paid run

# IBM credentials (once per machine)
python3 -c "from qiskit_ibm_runtime import QiskitRuntimeService; \
QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='YOUR_TOKEN')"
python3 -c "from qiskit_ibm_runtime import QiskitRuntimeService; \
print(QiskitRuntimeService().backend('ibm_fez').status())"
```

Exit criterion: the status call prints ibm_fez as operational with a queue count.

## Phase 1 — Tier 0 baseline sweep (free, hours)

Train all three ablation arms on the ideal simulator:

```bash
python -m qnn.train --tier tier0_ideal                    # Model A: hybrid
python -m qnn.train --tier tier0_ideal --model classical  # Model B: matched-param classical
python -m qnn.train --tier tier0_ideal --no-entangle      # Model C: product-state control
```

Each run writes `results_*.json` (loss curve, test accuracy, quantum grad-norm
telemetry) and a `.pt` checkpoint, plus `config_*.json` containing the SHA256
config fingerprint — commit that fingerprint file to the repo *before* Phase 4;
that is your preregistration artifact.

Watch for the barren-plateau warning (`|∇q| < 1e-3`). If it fires early and
persistently, drop `n_layers` from 3 to 2 in `config.py` and restart Phase 1.

Exit criteria: Model A converges; A ≳ C by a visible margin (entanglement is
doing something); A vs B gap recorded whichever way it goes.

## Phase 2 — Snapshot integration (free, ~1 hr)

Adapt your QPU Drift Collector export to the schema in `snapshot_template.json`
(field names/units are documented in `qnn/noise.py`). Then validate:

```bash
python -m qnn.preflight snapshot exports/fez_2026-07-12T0300.json
```

Fix anything flagged. Do this for at least 10–20 snapshots spanning several
days so Phase 3 sees calibration diversity.

## Phase 3 — Chain selection + transpile preflight (free, ~30 min)

```bash
# Score all 8-qubit heavy-hex paths across your snapshot archive
python -m qnn.preflight chains exports/*.json
# → prints e.g. "initial_layout = (43, 44, 45, 54, 64, 63, 62, 53)"
```

Put the winning chain into `QuantumConfig.initial_layout` in `qnn/config.py`,
then verify the compiled circuit is clean:

```bash
python -m qnn.preflight transpile --layout 43,44,45,54,64,63,62,53
```

Exit criterion: "✓ zero SWAP insertion". If SWAPs appear, the chain is not a
valid path in the live coupling map (an edge may be disabled today) — rerun
chain scoring with a fresh snapshot.

## Phase 4 — Tier 1 noisy validation (free, hours)

```bash
python -m qnn.train --tier tier1_noisy --snapshot exports/fez_2026-07-12T0300.json
```

Run this against 3–5 *different* snapshots. What you're measuring is the
Tier0→Tier1 accuracy drop and its variance across calibration states. If the
drop exceeds ~10 points, iterate here — deeper circuits, more shots, or
mitigation tuning — before spending hardware time. Nothing you learn on fez
will be interpretable if the sim-noise story isn't stable first.

Optional robustness variant: cycle a different snapshot per epoch
(noise-domain randomization) and see whether the Tier 2 transfer improves.

## Phase 5 — Tier 2 hardware run (PAID — the only paid phase)

Pre-run checklist: config fingerprint committed ✓ · lockfile pinned ✓ ·
preflight transpile clean *today* ✓ · fez queue acceptable ✓.

```bash
# 5a. Evaluation only (cheap: ~140 test circuits × 4096 shots)
python -m qnn.eval_hardware --weights results_tier0_ideal_hybrid.pt

# 5b. Optional SPSA fine-tune, same session (+100 circuits)
python -m qnn.eval_hardware --weights results_tier0_ideal_hybrid.pt --spsa
```

The script writes `tier2_calibration_stamp.json`. Immediately archive the drift
snapshot nearest that timestamp — you need it for Phase 6.

Budget note: 5a+5b together is under ~1M shots. Do NOT be tempted into full
gradient training on hardware; DESIGN §6 shows why (≈80M shots for a modest run).

## Phase 6 — Gap analysis & writeup (free)

Re-run Tier 1 against the *exact* calibration snapshot from the Phase 5 stamp
(post-hoc fair comparison), then assemble:

1. Accuracy table: {A, B, C} × {Tier 0, Tier 1, Tier 2}.
2. The Tier0→1→2 degradation curve for Model A, with Tier 1 variance bands
   from the multi-snapshot runs.
3. Gradient-norm trajectories (barren-plateau evidence).
4. Primary endpoint: Tier 2 hybrid (A) vs classical control (B).
   Secondary: A vs C on-hardware (does entanglement survive the noise?).

A null on the primary endpoint is a publishable result under this framing —
"calibration-seeded noise-domain randomization for sim-to-hardware QNN
transfer, with drift-robust qubit selection" stands on its own.

---

## Troubleshooting quick reference

**Runtime options warning from devices.py** — the plugin couldn't set
resilience/DD through internals; set them by constructing the Estimator options
manually per current `pennylane-qiskit` docs before the Phase 5 run.

**Tier 1 wildly slower than Tier 0** — expected; parameter-shift with shots is
~100× the cost of adjoint. Reduce epochs to ~10 for Tier 1; you're validating
robustness, not searching architectures.

**Accuracy collapse only on Tier 2** — first suspect readout: try
`resilience_level=2` on a small probe batch. Second suspect: the chain degraded
since scoring — re-run `preflight chains` including today's snapshot.

**Queue too long** — Sessions reserve the device once started; start the
session off-peak. Evaluation (5a) can run in batch mode instead if fine-tuning
is skipped.
