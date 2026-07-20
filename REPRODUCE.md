# Reproducing QNN-P1 results

## Environment

    python -m venv .venv
    .venv\Scripts\Activate.ps1        (Linux/Mac: source .venv/bin/activate)
    pip install -r requirements.lock

## Tier 0 (no IBM account needed, ~5 min CPU)

    python -m qnn.train --tier tier0_ideal --dataset qlabels --seed 8281

Expected: test accuracy at best-val = 0.785 (matches
results_tier0_ideal_hybrid_qlabels_s8281.json in this repo; the config
SHA256 printed at run start matches the committed config fingerprint).

## Tier 1 (drift-noise sim, ~10 hrs CPU, or reduce --epochs)

    python -m qnn.train --tier tier1_noisy --dataset qlabels --seed 8281 --shots 1024 --batch 16 --epochs 10 --snapshot exports\ibm_fez_2026-07-15T014954-0400_converted.json

Expected: TEST 0.7850 at the best-val checkpoint.

## Hardware (requires IBM Quantum account; ~8 min QPU per run)

    python -m qnn.eval_hardware_direct --weights results_tier1_noisy_hybrid_qlabels_s8281.pt --shots 1024

Measured 2026-07-19 on ibm_fez: 0.7850 (job d9eipdineu4c739ogic0).

Seeds fix all data splits; the quantum-labels task generator is seeded
independently (GEN_SEED=777 in qnn/datasets.py), so the task is identical
across machines.
