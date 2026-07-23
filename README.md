# QNN-P1: Drift-Archive Noise Models Predict Real Quantum Hardware

A preregistered study testing whether a noise model built from a
self-collected, month-long QPU calibration archive (hourly snapshots from a
Raspberry Pi 5 collector) can predict the behavior of a hybrid quantum
neural network on real IBM hardware. It can — to half a point.

**Full write-up:** [PAPER.md](PAPER.md)

![Sim vs hardware](figures/fig1_sim_vs_hardware.png)

## Headline results (all preregistered; see PREREGISTRATION.md)

| Endpoint | Criterion | Result |
|---|---|---|
| P1 | Hardware A in [0.72, 0.85]; vs classical B = 0.7385 | 0.7850 — passed; +4.7 pts over classical |
| P2 | Hardware A-C separation <= 0.05 | 0.030 — passed, matching sim prediction |
| P3 | Sim-to-hardware gap (run-day calibration) | A: +0.5 pt, C: +0.5 pt |

## Full results (seed 8281 unless noted)

| Stage | Model A (entangled) | Model C (no entangle) | Model B (classical) |
|---|---|---|---|
| Tier 0 ideal sim (n=10, mean +/- sd) | 0.739 +/- 0.028 | 0.745 +/- 0.037 | 0.739 +/- 0.053 |
| Tier 1 sim — worst day (chain) | 0.7850 | 0.7550 | — |
| Tier 1 sim — best day (chain) | 0.7700 | — | — |
| Run-day sim (Jul 19 calibration) | 0.7800 | 0.7500 | — |
| **ibm_fez hardware (measured)** | **0.7850** | **0.7550** | — |

Model B (classical control) is noise-free by construction; its Tier 0 value
is the reference across tiers. Tier 0 is the only multi-seed cell (n=10);
hardware cells are single runs (~1 pt measurement noise on 200 samples).
The A-C separation is 3.0 pts in simulation and 3.0 pts on hardware — the
structure transferred, not just the magnitudes.

![Tier progression](figures/fig3_tier_progression.png)

## Two additional findings

**Chain selection dominates noise.** Same model, seed, and calibration
snapshot: 0.7350 on a generic qubit region (0-7) vs 0.7850 on the
drift-selected chain (125-140). Chain selection from the archive recovered
the entire noise degradation — a +5.0-pt controlled, single-variable effect.

![Chain effect](figures/fig2_chain_effect.png)

**Calibration timing is null on a good chain.** An 89% spread in
chain-region composite error across the archive produced only a 1.5-pt
inverted accuracy difference (worst day 0.7850 > best day 0.7700). Where
you sit on the lattice matters; when you run does not, once the chain is
well-chosen. The archive's demonstrated value is layout selection, not
day-picking.

## Method in one paragraph

A 3-arm ablation (A: 8-qubit entangled VQC hybrid; B: parameter-matched
classical MLP; C: entanglement-free hybrid) trained on a quantum-generated
labeling task, evaluated across three tiers: ideal simulation (10 seeds per
arm), drift-seeded noisy simulation (Aer noise models built from archived
ibm_fez calibration snapshots, remapped onto the selected physical chain),
and real ibm_fez hardware via a single-job EstimatorV2 evaluation. The
qubit chain was selected by scoring all 950 8-qubit heavy-hex paths against
20 daily snapshots; the winner (125-124-123-136-143-142-141-140) transpiles
with zero SWAP insertion (depth 21, 21 CZ). Comparisons are paired by seed;
the null Tier 0 result (all three arms tie) is what makes the noise-response
tiers interpretable.

## Honest failure log

This study kept a live deviations log (see PREREGISTRATION.md). Notable:
hardware cost was initially misestimated by an order of magnitude, two
hardware runs were incorrectly cancelled based on a stale quota warning and
a flawed cost model, and shot count was amended 4096 -> 1024 after committed
reevals demonstrated precision saturation. All deviations were logged
before unblinding any result. The failures are part of the record because
methods that only report successes are not methods.

## Part of the IRMB program

QNN-P1 belongs to a longer experimental sequence examining quantum-generated
coordination signals and the conditions required to interpret them
responsibly:

- [Design 3 — Bell Coordination](https://github.com/billyrdavis1985-bot/IRMB-Phase7G-Design3-BellCoordination):
  CHSH testing in multi-agent LLM systems modulated by real entanglement
  (ibm_fez + IonQ Forte-1 + 7-model council). Primary CHSH result null;
  quantum measurement distributions statistically distinguishable from
  sham controls.
- [Design 4 — Quantum Coordination](https://github.com/billyrdavis1985-bot/IRMB_Phase7G_Design4_QuantumCoordination):
  300 contested scientific claims across QPU, PRNG, and emulator
  conditions. Strong distributional divergence, no aggregate QPU
  performance advantage.
- [Design 5 — Quantum Causality](https://github.com/billyrdavis1985-bot/IRMB_Phase7G_Design5_QuantumCausality):
  matched-distribution causal control; the real-IBM condition reversed
  (C1 F1 = 0.6136 vs matched control 0.7767) under overlapping confounds
  including QPU decoherence; |S| = 0.0120, no Bell violation.
  ([dataset build](https://github.com/billyrdavis1985-bot/IRMB_Design5_Dataset_MultiAgent_Build))

Design 5 identified QPU decoherence and calibration drift as confounds it
could not isolate. QNN-P1 turned that same drift into a predictive
instrument: the [QPU Drift Collector](https://github.com/billyrdavis1985-bot/qpu-drift-collector)
(Raspberry Pi 5, hourly ibm_fez / ibm_marrakesh calibration telemetry)
supplied the 389-snapshot archive used here for chain selection and noise
modeling. A confound identified in one study became the measurement target
of the next.

Next: a drift-aware quantum error correction line (QEC-P1), then Design 6
with hardware-qualification gating. See PAPER.md section 7.

## Repository structure

    IRMB_QNN-P1_DriftArchive_HardwareValidation/
    ├── README.md                 <- you are here
    ├── PAPER.md                  <- full write-up
    ├── PREREGISTRATION.md        <- endpoints, thresholds, live deviations log
    ├── DESIGN.md                 <- three-tier protocol design
    ├── REPRODUCE.md              <- how to re-run each tier
    ├── RUNBOOK.md                <- operational procedure
    ├── LICENSE                   <- MIT
    ├── qnn/                      <- the package (models, noise, training, hardware eval)
    ├── figures/                  <- fig1-fig3
    ├── best-day/                 <- best-day calibration reeval
    ├── runs/                     <- immutable hardware run evidence
    ├── requirements.txt          <- dependencies
    ├── requirements.lock         <- pinned versions
    ├── snapshot_template.json    <- calibration snapshot schema
    ├── hw_job_id.json            <- hardware job identifiers
    └── run_sweep.ps1             <- sweep driver

## Reproduce

See REPRODUCE.md. Tier 0 reproduces on any machine in ~5 minutes with no
IBM account. Hardware job IDs and calibration timestamps are preserved in
runs/ and hw_job_id.json.

## License

MIT (code). Independent Research in Multi-agent Benchmarking (IRMB),
Hudson Forge Technologies LLC — self-funded.
