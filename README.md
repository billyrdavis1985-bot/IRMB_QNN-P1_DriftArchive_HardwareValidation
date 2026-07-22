# QNN-P1: Drift-Archive Noise Modeling and Hardware Validation

A preregistered pilot study testing whether noise models built from a self-collected, month-long QPU calibration archive can approximate the behavior of a hybrid quantum neural network on real IBM hardware.

The calibration archive was produced through hourly snapshots collected by a Raspberry Pi 5. Within this pilot, run-day simulation and measured hardware accuracy differed by **0.5 percentage points for both evaluated quantum arms**.

These are single-run hardware-validation results. They support the feasibility of the collector-to-noise-model-to-hardware workflow but do not, by themselves, establish general predictive validity or quantum advantage.

![Simulation versus hardware](figures/fig1_sim_vs_hardware.png)

## Project scope

QNN-P1 was designed as a calibration and methods-development study supporting future IRMB QEC and Design 6 experiments.

Its primary goals were to:

* Validate the calibration-archive-to-noise-model pipeline.
* Test hardware-aware qubit-chain selection.
* Measure simulation-to-hardware transfer.
* Exercise preregistration and deviation logging.
* Identify execution, cost, and evidence-preservation requirements before larger studies.

## Preregistered endpoints

| Endpoint | Preregistered criterion                        | Observed result             |
| -------- | ---------------------------------------------- | --------------------------- |
| P1       | Hardware A within `[0.72, 0.85]`               | `0.7850` — criterion passed |
| P2       | Hardware A–C separation `≤ 0.05`               | `0.030` — criterion passed  |
| P3       | Measure the run-day simulation-to-hardware gap | A: `0.005`; C: `0.005`      |

Model A exceeded the Tier 0 classical control by approximately 4.7 percentage points in this evaluation. That comparison is descriptive; the pilot was not powered to establish a general quantum performance advantage.

## Full results

Seed 8281 unless otherwise noted.

| Stage                                                 | Model A: entangled hybrid | Model C: no-entanglement hybrid | Model B: classical control |
| ----------------------------------------------------- | ------------------------: | ------------------------------: | -------------------------: |
| Tier 0 ideal simulation, `n=10`, mean ± σ             |           `0.739 ± 0.028` |                 `0.745 ± 0.037` |            `0.739 ± 0.053` |
| Tier 1 simulation, selected chain, worst archived day |                  `0.7850` |                        `0.7550` |                          — |
| Tier 1 simulation, selected chain, best archived day  |                  `0.7700` |                               — |                          — |
| Run-day simulation, July 19 calibration               |                  `0.7800` |                        `0.7500` |                          — |
| **ibm_fez hardware**                                  |                **0.7850** |                      **0.7550** |                          — |

Model B is noise-free by construction, so its Tier 0 result is used as the classical reference across tiers.

Tier 0 is the only multi-seed stage. Each hardware cell represents one 200-sample evaluation at 1,024 shots. The observed simulation-to-hardware match should therefore be interpreted as pilot evidence rather than a population-level performance estimate.

The A–C separation was 3.0 percentage points in both the run-day simulation and the hardware evaluation. This shows that the observed relationship between the two evaluated arms transferred alongside their aggregate accuracies.

![Tier progression](figures/fig3_tier_progression.png)

## Additional findings

### Qubit-chain selection produced a five-point simulated effect

Using the same model, training seed, and calibration snapshot:

* Generic physical region `0–7`: `0.7350`
* Drift-selected physical path: `0.7850`

The selected path was:

```text
[125, 124, 123, 136, 143, 142, 141, 140]
```

Under this controlled noisy-simulation comparison, physical-chain selection produced a **5.0-percentage-point accuracy difference**.

This result supports hardware-aware placement as an important experimental variable. It does not yet establish that the same effect size would reproduce across seeds, tasks, snapshots, or matched hardware runs.

![Chain-selection effect](figures/fig2_chain_effect.png)

### No calibration-timing effect was detected within the pilot

The selected chain experienced an 89% spread in its composite calibration score across the archived period. The measured accuracy difference between the selected best-day and worst-day simulations was only 1.5 percentage points and occurred in the opposite direction from the raw calibration ranking.

Within the tested conditions, calibration-date variation did not produce a clear performance effect once the physical chain had been selected.

This is a pilot null result, not proof that calibration timing is irrelevant. A stronger conclusion would require more seeds, snapshots, repeated hardware runs, and a preregistered equivalence bound.

## Method

The study used a three-arm ablation:

* **Model A:** eight-qubit entangled variational quantum-circuit hybrid.
* **Model B:** classical control using the same encoder and head with an approximately parameter-matched MLP replacing the quantum layer.
* **Model C:** hybrid quantum model with the entangling CZ gates removed.

The models were trained on a fixed quantum-generated binary-labeling task and evaluated across three execution tiers:

1. **Tier 0 — ideal simulation:** ten seeds per arm.
2. **Tier 1 — archived-calibration noisy simulation:** Qiskit Aer noise models reconstructed from archived `ibm_fez` calibration snapshots.
3. **Tier 2 — hardware validation:** forward-pass evaluation on `ibm_fez` using a direct single-job `EstimatorV2` execution path.

The archived calibration data were remapped onto the selected physical chain so that the Tier 1 noise model represented the qubits used during the hardware evaluation.

The chain-selection procedure scored 950 candidate eight-qubit heavy-hex paths across 20 archived daily snapshots. The selected path transpiled with zero SWAP insertion at depth 21 with 21 CZ gates.

The Tier 0 tie between the three arms provides the baseline needed to interpret later noise-response and hardware-transfer differences.

## Evidence records

The completed hardware evaluations are preserved as separate run records:

* [`runs/2026-07-19_A_entangled_s8281`](runs/2026-07-19_A_entangled_s8281)
* [`runs/2026-07-19_C_no-entangle_s8281`](runs/2026-07-19_C_no-entangle_s8281)

Each run record includes available configuration, simulation, hardware, and checkpoint-identity artifacts.

The repository also includes:

* Hardware job IDs
* Calibration timestamps
* Shot counts
* Mitigation settings
* Checkpoint SHA-256 fingerprints
* Source commit references
* Tier 1 reevaluation results
* An artifact hash inventory

The model checkpoint binaries are not committed to standard Git history. Their identities are preserved through SHA-256 fingerprints.

## Deviations and failure record

The study maintained a live deviations log in [`PREREGISTRATION.md`](PREREGISTRATION.md).

Notable deviations included:

* Initial hardware cost was underestimated.
* Two early hardware attempts were cancelled based on an incorrect quota interpretation and flawed cost assumptions.
* The PennyLane hardware path created one Runtime job per circuit, making it economically impractical under the available plan.
* The hardware evaluator was replaced with a direct `EstimatorV2` single-job execution path.
* Hardware shots were reduced from 4,096 to 1,024 after committed reevaluations indicated precision saturation.
* Planned repeated hardware runs were cancelled after the corrected cost model was established.

The endpoint definitions and comparison anchors were not changed after hardware results were unblinded.

The failure record is retained because the execution constraints and corrective decisions are part of the experimental method.

## Reproduction

See [`REPRODUCE.md`](REPRODUCE.md) for environment setup and execution commands.

Tier 0 can be reproduced locally without an IBM Quantum account. Tier 1 requires the appropriate archived calibration snapshot. Hardware reproduction requires IBM Quantum access and may be affected by backend availability, calibration drift, queue conditions, API changes, and account limits.

## Interpretation boundary

This pilot supports the following conclusions:

* Archived QPU calibration telemetry can be converted into reproducible first-order noise models.
* Archive-informed physical-chain selection can materially affect simulated performance.
* The evaluated run-day noise models closely matched two hardware accuracy measurements.
* A direct, batched Runtime execution path substantially reduced hardware overhead.
* Preregistration, deviation logging, and immutable evidence records are practical for small independent quantum experiments.

This pilot does **not** establish:

* General quantum advantage over classical models.
* Universal simulation-to-hardware prediction accuracy.
* A causal entanglement advantage.
* Calibration timing irrelevance across devices or tasks.
* Generalization beyond the tested task, seed, backend, architecture, and physical chain.

* ## Repository map

### Experimental implementation

- [`qnn/train.py`](qnn/train.py) — Tier 0 and Tier 1 training
- [`qnn/model.py`](qnn/model.py) — Models A, B, and C
- [`qnn/circuits.py`](qnn/circuits.py) — variational quantum circuit
- [`qnn/noise.py`](qnn/noise.py) — archive-derived noise models and chain scoring
- [`qnn/eval_hardware_direct.py`](qnn/eval_hardware_direct.py) — direct IBM Runtime hardware evaluation
- [`qnn/preflight.py`](qnn/preflight.py) — snapshot, chain, and transpilation checks
- [`qnn/config.py`](qnn/config.py) — experiment configuration and fingerprints
- [`qnn/datasets.py`](qnn/datasets.py) — fixed quantum-label dataset generator

### Research records

- [`PREREGISTRATION.md`](PREREGISTRATION.md) — locked endpoints and deviations
- [`REPRODUCE.md`](REPRODUCE.md) — reproduction commands
- [`DESIGN.md`](DESIGN.md) — architecture and experimental rationale
- [`runs/`](runs/) — immutable A and C hardware evidence packages

- ### Tested environment

- Python 3.11.9
- Windows 10 and PowerShell
- Qiskit 2.3.0
- Qiskit Aer 0.17.2
- Qiskit IBM Runtime 0.45.1
- PennyLane 0.45.1
- PyTorch 2.10.0

## IRMB program context

QNN-P1 shares infrastructure and methodology with the IRMB quantum-causality research line:

* [IRMB Phase 7G Design 5 — Quantum Causality](https://github.com/billyrdavis1985-bot/IRMB_Phase7G_Design5_QuantumCausality)
* [IRMB Design 5 — Multi-Agent Dataset Build](https://github.com/billyrdavis1985-bot/IRMB_Design5_Dataset_MultiAgent_Build)

The [QPU Drift Collector](https://github.com/billyrdavis1985-bot/qpu-drift-collector) continuously collects hourly calibration telemetry from `ibm_fez` and `ibm_marrakesh`.

The 389-snapshot archive used for noise modeling and chain-selection analysis in this pilot was produced by that collector.

## License

MIT License for code.

Independent Research in Multi-Agent Benchmarking — IRMB
Hudson Forge Technologies LLC
Self-funded independent research
﻿
