"""Direct-EstimatorV2 hardware evaluation. ONE Runtime job for the whole test set.

Why this exists: the pennylane-qiskit plugin submits one Runtime job per
circuit (per sample). On Open plan, per-job billed overhead (~30s) makes that
path economically impossible (200 samples -> ~100 QPU-minutes). This script
bypasses PennyLane on hardware: it computes the encoder angles classically,
binds the trained quantum weights into a single parameterized ISA circuit,
and submits ONE EstimatorV2 job containing 8 pubs (one per <Z_i>) x 200
parameter sets. Expected QPU cost: one job, minutes not tens of minutes.

Safety: the job_id is written to disk IMMEDIATELY after submission. If the
local script dies or is interrupted, the job keeps running server-side and
results are retrieved later with --retrieve JOB_ID at zero additional cost.

Usage:
  # free local dress rehearsal (StatevectorEstimator, no account needed):
  python -m qnn.eval_hardware_direct --weights <ckpt.pt> --dry-run

  # real submission:
  python -m qnn.eval_hardware_direct --weights <ckpt.pt>

  # recover a previous submission:
  python -m qnn.eval_hardware_direct --weights <ckpt.pt> --retrieve <JOB_ID>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import torch

from .config import DEFAULT
from .train import load_data
from .model import Encoder


def build_circuit(n_qubits: int, n_layers: int, weights: np.ndarray,
                  entangle: bool):
    """Parameterized circuit: 8 free input Parameters (encoder angles),
    trained variational weights bound as constants. Mirrors qnn.circuits."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    x = [Parameter(f"x{i}") for i in range(n_qubits)]
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(x[i], i)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.ry(float(weights[layer, i, 0]), i)
            qc.rz(float(weights[layer, i, 1]), i)
        if entangle:
            for i in range(0, n_qubits - 1, 2):
                qc.cz(i, i + 1)
            for i in range(1, n_qubits - 1, 2):
                qc.cz(i, i + 1)
    return qc, x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--no-entangle", action="store_true")
    p.add_argument("--seed", type=int, default=8281)
    p.add_argument("--dataset", choices=["cancer", "qlabels"], default="qlabels")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="local StatevectorEstimator; no IBM account, no quota")
    p.add_argument("--retrieve", type=str, default=None,
                   help="skip submission; fetch results for this job id")
    a = p.parse_args()

    cfg = dataclasses.replace(
        DEFAULT, train=dataclasses.replace(DEFAULT.train, seed=a.seed))
    _, test_ds, cfg = load_data(cfg, a.dataset)
    n = cfg.quantum.n_qubits

    # --- classical halves of the model, run locally ---
    state = torch.load(a.weights)
    enc = Encoder(cfg)
    enc.load_state_dict({k.replace("encoder.", ""): v for k, v in state.items()
                         if k.startswith("encoder.")})
    enc.eval()
    W = state["qlayer.weights"].numpy()          # (layers, qubits, 2)
    head_w = state["head.weight"].numpy()        # (classes, qubits)
    head_b = state["head.bias"].numpy()

    X = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    y = np.array([int(test_ds[i][1]) for i in range(len(test_ds))])
    if a.limit:
        X, y = X[: a.limit], y[: a.limit]
    with torch.no_grad():
        angles = enc(X).numpy()                  # (N, n_qubits)
    N = len(angles)

    qc, x_params = build_circuit(n, cfg.quantum.n_layers, W,
                                 entangle=not a.no_entangle)
    from qiskit.quantum_info import SparsePauliOp
    observables = [SparsePauliOp("I" * (n - 1 - i) + "Z" + "I" * i)
                   for i in range(n)]            # Z on qubit i (little-endian)
    param_order = list(qc.parameters)            # sorted; map columns to it
    col = {f"x{i}": i for i in range(n)}
    pv = np.stack([angles[:, col[pr.name]] for pr in param_order], axis=1)

    stamp = {"wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "weights": a.weights, "entangle": not a.no_entangle,
             "seed": a.seed, "shots": a.shots, "n_samples": N,
             "mode": "dry-run" if a.dry_run else "hardware"}

    if a.dry_run:
        from qiskit.primitives import StatevectorEstimator
        est = StatevectorEstimator()
        pubs = [(qc, obs, pv) for obs in observables]
        result = est.run(pubs).result()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        service = QiskitRuntimeService()
        backend = service.backend(cfg.hardware.backend_name)
        layout = list(cfg.quantum.initial_layout)
        pm = generate_preset_pass_manager(optimization_level=3,
                                          backend=backend,
                                          initial_layout=layout)
        isa = pm.run(qc)
        iso_obs = [obs.apply_layout(isa.layout) for obs in observables]
        est = EstimatorV2(mode=backend)
        est.options.default_shots = a.shots
        try:
            est.options.resilience_level = 1
            est.options.dynamical_decoupling.enable = True
            est.options.dynamical_decoupling.sequence_type = "XpXm"
            stamp["mitigation"] = "resilience=1, DD=XpXm (set directly)"
        except Exception as e:  # noqa: BLE001
            stamp["mitigation"] = f"defaults (options failed: {e})"

        if a.retrieve:
            job = service.job(a.retrieve)
            print(f"retrieving job {a.retrieve} ({job.status()})", flush=True)
        else:
            pubs = [(isa, obs, pv) for obs in iso_obs]
            job = est.run(pubs)
            stamp["job_id"] = job.job_id()
            with open("hw_job_id.json", "w") as f:      # crash-proof FIRST
                json.dump(stamp, f, indent=2)
            print(f"SUBMITTED single job: {job.job_id()}", flush=True)
            print("job id saved to hw_job_id.json - Ctrl+C is now SAFE; "
                  "recover anytime with --retrieve", flush=True)
        try:
            props = backend.properties()
            stamp["calibration_time"] = str(getattr(props, "last_update_date", "n/a"))
        except Exception:
            stamp["calibration_time"] = "unavailable"
        result = job.result()

    evs = np.stack([result[i].data.evs for i in range(n)], axis=1)  # (N, n)
    logits = evs @ head_w.T + head_b
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    stamp["test_acc"] = acc

    tag = "dryrun" if a.dry_run else "hardware"
    out = a.weights.split("\\")[-1].split("/")[-1].replace(
        ".pt", f"_direct_{tag}.json")
    with open(out, "w") as f:
        json.dump(stamp, f, indent=2)
    print(f"\n{tag.upper()} accuracy ({N} samples, {a.shots} shots): "
          f"{acc:.4f}  ->  {out}")


if __name__ == "__main__":
    main()
