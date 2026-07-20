"""Re-evaluate a trained checkpoint at full measurement precision.

Why this exists: Tier 1 *training* runs at reduced shots (e.g. 1024) for
tractability, but tier-to-tier comparisons must be measured at the same
precision Tier 2 will use (4096). This loads a checkpoint, rebuilds the
noisy device at full shots, and scores the untouched test set once.

Usage:
    python -m qnn.reeval --weights results_tier1_noisy_hybrid_qlabels_s8281.pt ^
        --snapshot exports\ibm_fez_2026-07-15T014954-0400_converted.json
    # add --no-entangle for Model C checkpoints; --seed must match the run
"""
from __future__ import annotations

import argparse
import dataclasses
import json

import torch
from torch.utils.data import DataLoader

from .config import DEFAULT, Tier
from .devices import make_device
from .model import HybridQNN
from .train import load_data, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--dataset", choices=["cancer", "qlabels"], default="qlabels")
    p.add_argument("--seed", type=int, default=8281,
                   help="MUST match the training run (fixes the test split)")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--no-entangle", action="store_true")
    a = p.parse_args()

    cfg = dataclasses.replace(
        DEFAULT,
        train=dataclasses.replace(DEFAULT.train, seed=a.seed),
        quantum=dataclasses.replace(DEFAULT.quantum, shots=a.shots),
        hardware=dataclasses.replace(DEFAULT.hardware,
                                     calibration_snapshot=a.snapshot),
    )
    _, test_ds, cfg = load_data(cfg, a.dataset)
    test_dl = DataLoader(test_ds, batch_size=64)

    device = make_device(Tier.NOISY_SIM, cfg)
    model = HybridQNN(cfg, device, entangle=not a.no_entangle,
                      diff_method="parameter-shift", per_sample=True)
    model.load_state_dict(torch.load(a.weights))

    acc = evaluate(model, test_dl)
    out = a.weights.replace(".pt", f"_reeval{a.shots}.json")
    with open(out, "w") as f:
        json.dump({"weights": a.weights, "snapshot": a.snapshot,
                   "shots": a.shots, "seed": a.seed,
                   "entangle": not a.no_entangle,
                   "test_acc_fullshot": acc}, f, indent=2)
    print(f"test accuracy @ {a.shots} shots: {acc:.4f}  →  {out}")


if __name__ == "__main__":
    main()
