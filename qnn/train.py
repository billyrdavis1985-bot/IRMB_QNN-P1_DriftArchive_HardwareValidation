"""Tier 0 / Tier 1 training entry point.

Usage:
    python -m qnn.train --tier tier0_ideal
    python -m qnn.train --tier tier1_noisy --snapshot path/to/drift_snapshot.json
    python -m qnn.train --tier tier0_ideal --model classical   # Model B baseline
    python -m qnn.train --tier tier0_ideal --no-entangle       # Model C control
"""
from __future__ import annotations

import argparse
import re
import dataclasses
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DEFAULT, ExperimentConfig, Tier
from .devices import make_device
from .model import HybridQNN, ClassicalControl


def load_data(cfg: ExperimentConfig, dataset: str = "cancer"):
    from . import datasets
    train, test, input_dim = datasets.load(dataset, cfg, cfg.train.seed)
    if input_dim != cfg.classical.input_dim:
        cfg = dataclasses.replace(
            cfg, classical=dataclasses.replace(cfg.classical, input_dim=input_dim))
    return train, test, cfg


@torch.no_grad()
def evaluate(model, loader) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum())
        total += len(yb)
    return correct / total


def train(cfg: ExperimentConfig, tier: Tier, model_kind: str, entangle: bool,
          dataset: str = "cancer"):
    torch.manual_seed(cfg.train.seed)
    train_ds, test_ds, cfg = load_data(cfg, dataset)
    # Carve a validation split from TRAIN (test stays untouched until the end).
    n_val = max(32, int(0.15 * len(train_ds)))
    gen = torch.Generator().manual_seed(cfg.train.seed)
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [len(train_ds) - n_val, n_val], generator=gen)
    train_dl = DataLoader(train_sub, batch_size=cfg.train.batch_size, shuffle=True)
    val_dl = DataLoader(val_sub, batch_size=256)
    test_dl = DataLoader(test_ds, batch_size=256)

    if model_kind == "classical":
        model = ClassicalControl(cfg)
    else:
        diff = "adjoint" if tier is Tier.IDEAL else "parameter-shift"
        device = make_device(tier, cfg)
        model = HybridQNN(cfg, device, entangle=entangle, diff_method=diff,
                          per_sample=(tier is Tier.NOISY_SIM))

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    loss_fn = nn.CrossEntropyLoss()
    import os as _os
    _snap = cfg.hardware.calibration_snapshot
    snap_tag = ""
    if _snap:
        digits = re.sub(r"[^0-9]", "", _os.path.basename(_snap))
        snap_tag = "_snap" + digits[:8] if digits else ""
    log = {"config_sha256": cfg.fingerprint(), "tier": tier.value,
           "model": model_kind, "entangle": entangle, "dataset": dataset,
           "snapshot": _os.path.basename(_snap) if _snap else None,
           "history": []}

    import copy
    best_val, best_epoch, best_state = -1.0, -1, None
    step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        t0 = time.time()
        batch_log_every = 1 if tier is not Tier.IDEAL else 0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()

            if step % cfg.train.grad_norm_log_every == 0 and hasattr(model, "quantum_grad_norm"):
                gn = model.quantum_grad_norm()
                log["history"].append({"step": step, "loss": loss.item(),
                                       "q_grad_norm": gn})
                if gn == gn and gn < 1e-3 and step > 50 and loss.item() > 0.1:   # nan-safe check
                    print(f"⚠ barren-plateau warning: |∇q| = {gn:.2e} at step {step}")
            opt.step()
            step += 1
            if step % max(1, batch_log_every) == 0:
                print(f"  batch {step:04d}  loss {loss.item():.4f}  "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)

        val_acc = evaluate(model, val_dl)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:02d}  loss {loss.item():.4f}  "
              f"val_acc {val_acc:.4f}{'*' if best_epoch == epoch else ' '} "
              f"({time.time()-t0:.1f}s)")
        log["history"].append({"epoch": epoch, "val_acc": val_acc})

    # Restore the best-validation checkpoint; touch test exactly once.
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_dl)
    log.update({"seed": cfg.train.seed, "best_epoch": best_epoch,
                "best_val_acc": best_val, "test_acc_at_best_val": test_acc})
    out = (f"results_{tier.value}_{model_kind}"
           f"{'_noent' if not entangle else ''}_{dataset}{snap_tag}_s{cfg.train.seed}.json")
    with open(out, "w") as f:
        json.dump(log, f, indent=2)
    print(f"best val {best_val:.4f} @ epoch {best_epoch}  →  "
          f"TEST {test_acc:.4f}  →  {out}")
    torch.save(model.state_dict(), out.replace(".json", ".pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=Tier, choices=list(Tier), default=Tier.IDEAL)
    p.add_argument("--model", choices=["hybrid", "classical"], default="hybrid")
    p.add_argument("--no-entangle", action="store_true")
    p.add_argument("--dataset", choices=["cancer", "qlabels"], default="cancer")
    p.add_argument("--seed", type=int, default=None, help="override TrainConfig.seed")
    p.add_argument("--shots", type=int, default=None, help="override QuantumConfig.shots")
    p.add_argument("--batch", type=int, default=None, help="override TrainConfig.batch_size")
    p.add_argument("--epochs", type=int, default=None, help="override TrainConfig.epochs")
    p.add_argument("--snapshot", type=str, default=None,
                   help="drift-collector calibration JSON for Tier 1")
    a = p.parse_args()

    cfg = DEFAULT
    if a.seed is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seed=a.seed))
    if a.shots is not None:
        cfg = dataclasses.replace(cfg, quantum=dataclasses.replace(cfg.quantum, shots=a.shots))
    if a.batch is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, batch_size=a.batch))
    if a.epochs is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, epochs=a.epochs))
    if a.snapshot:
        cfg = dataclasses.replace(
            cfg, hardware=dataclasses.replace(cfg.hardware,
                                              calibration_snapshot=a.snapshot))
    cfg.dump(f"config_{a.tier.value}.json")   # preregister before running
    train(cfg, a.tier, a.model, entangle=not a.no_entangle, dataset=a.dataset)
