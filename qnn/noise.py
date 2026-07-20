"""Tier 1 noise models, seeded from real ibm_fez calibration data.

Two paths:
  1. Live:      NoiseModel.from_backend(ibm_fez)  — one frozen snapshot.
  2. Archived:  built from a QPU Drift Collector JSON snapshot — lets you
                train against *sampled* calibration states over time
                (noise-domain randomization) and reproduce the exact
                snapshot used in a past run.

The archived path assumes the drift collector stores, per snapshot:
  { "timestamp": ..., "qubits": {q: {"T1_us", "T2_us", "readout_error"}},
    "gates": {"cz": {"(q1,q2)": err}, "sx": {q: err}, "x": {q: err}} }
Adapt FIELD_MAP if your schema differs.
"""
from __future__ import annotations

import json


def build_noise_model(cfg):
    from qiskit_aer.noise import NoiseModel

    snapshot_path = cfg.hardware.calibration_snapshot
    if snapshot_path is None:
        return _from_live_backend(cfg)
    chain = getattr(cfg.quantum, "initial_layout", None)
    if chain is not None:
        return _from_drift_snapshot(_remap_to_chain(snapshot_path, chain), cfg)
    return _from_drift_snapshot(snapshot_path, cfg)


def _from_live_backend(cfg):
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_aer.noise import NoiseModel

    service = QiskitRuntimeService()
    backend = service.backend(cfg.hardware.backend_name)
    return NoiseModel.from_backend(backend)


def _from_drift_snapshot(path: str, cfg):
    """Reconstruct a NoiseModel from archived drift-collector telemetry.

    Models included: depolarizing error on sx/x/cz scaled to reported gate
    error, thermal relaxation from T1/T2 at nominal gate durations, and
    per-qubit readout error. This is deliberately a *first-order* model —
    it will not capture crosstalk or non-Markovian drift, which is exactly
    the residual the Tier1→Tier2 gap analysis is designed to expose.
    """
    from qiskit_aer.noise import (
        NoiseModel, depolarizing_error, thermal_relaxation_error,
        ReadoutError,
    )

    with open(path) as f:
        snap = json.load(f)

    # Nominal Heron r2 gate durations (seconds); refine from backend.target
    # properties if your collector archives them.
    T_1Q, T_2Q = 60e-9, 84e-9

    nm = NoiseModel(basis_gates=["cz", "id", "rz", "sx", "x"])

    for q_str, props in snap["qubits"].items():
        q = int(q_str)
        t1 = props["T1_us"] * 1e-6
        t2 = min(props["T2_us"] * 1e-6, 2 * t1)  # enforce T2 <= 2*T1

        relax_1q = thermal_relaxation_error(t1, t2, T_1Q)
        for gate in ("sx", "x"):
            g_err = snap.get("gates", {}).get(gate, {}).get(q_str)
            err = relax_1q
            if g_err:
                err = err.compose(depolarizing_error(g_err, 1))
            nm.add_quantum_error(err, gate, [q])

        ro = props.get("readout_error")
        if ro is not None:
            nm.add_readout_error(
                ReadoutError([[1 - ro, ro], [ro, 1 - ro]]), [q]
            )

    for pair_str, g_err in snap.get("gates", {}).get("cz", {}).items():
        q1, q2 = (int(x) for x in pair_str.strip("()").split(","))
        # skip edges touching uncalibrated qubits (e.g. fez qubit 72)
        if str(q1) not in snap["qubits"] or str(q2) not in snap["qubits"]:
            continue
        t1a = snap["qubits"][str(q1)]["T1_us"] * 1e-6
        t2a = min(snap["qubits"][str(q1)]["T2_us"] * 1e-6, 2 * t1a)
        t1b = snap["qubits"][str(q2)]["T1_us"] * 1e-6
        t2b = min(snap["qubits"][str(q2)]["T2_us"] * 1e-6, 2 * t1b)
        relax_2q = thermal_relaxation_error(t1a, t2a, T_2Q).tensor(
            thermal_relaxation_error(t1b, t2b, T_2Q))
        err = relax_2q.compose(depolarizing_error(g_err, 2))
        nm.add_quantum_error(err, "cz", [q1, q2])

    return nm


def score_linear_chains(snapshot_paths: list[str], coupling_edges: list[tuple[int, int]],
                        chain_len: int = 8) -> list[tuple[float, tuple[int, ...]]]:
    """Drift-robust qubit-chain selection (DESIGN §4).

    Scores every simple path of length `chain_len` in the coupling graph by
    mean composite error across ALL provided snapshots, so chains that are
    only momentarily good get penalized. Returns chains sorted best-first;
    feed the winner into QuantumConfig.initial_layout.
    """
    import itertools, collections

    adj = collections.defaultdict(set)
    for a, b in coupling_edges:
        adj[a].add(b); adj[b].add(a)

    snaps = []
    for p in snapshot_paths:
        with open(p) as f:
            snaps.append(json.load(f))

    def chain_cost(chain) -> float:
        cost = 0.0
        for snap in snaps:
            # skip chains touching qubits absent from calibration
            if any(str(q) not in snap["qubits"] for q in chain):
                return float("inf")
            for q in chain:
                qp = snap["qubits"][str(q)]
                cost += qp.get("readout_error", 0.01)
                cost += 1.0 / max(qp["T1_us"], 1.0)          # coherence penalty
            for a, b in zip(chain, chain[1:]):
                cz = snap.get("gates", {}).get("cz", {})
                cost += cz.get(f"({a},{b})", cz.get(f"({b},{a})", 0.02)) * 10
        return cost / len(snaps)

    # DFS enumeration of simple paths (fine at heavy-hex degree <= 3)
    paths = []
    def dfs(path):
        if len(path) == chain_len:
            paths.append(tuple(path)); return
        for nxt in adj[path[-1]]:
            if nxt not in path:
                dfs(path + [nxt])
    for start in adj:
        dfs([start])

    dedup = {tuple(min(p, p[::-1])) for p in paths}
    scored = sorted((chain_cost(c), c) for c in dedup)
    return scored


def _remap_to_chain(path: str, chain) -> str:
    """Write a temp snapshot whose logical qubits 0..n-1 carry the calibration
    of the selected physical chain, so Aer simulates the qubits the hardware
    run will actually use."""
    import json, tempfile, os
    with open(path) as f:
        snap = json.load(f)
    out = {"timestamp": snap.get("timestamp"), "backend": snap.get("backend"),
           "remapped_from_chain": list(chain),
           "qubits": {}, "gates": {"sx": {}, "x": {}, "cz": {}}}
    for i, p in enumerate(chain):
        qp = snap["qubits"].get(str(p))
        if qp is not None:
            out["qubits"][str(i)] = qp
        for g in ("sx", "x"):
            e = snap.get("gates", {}).get(g, {}).get(str(p))
            if e is not None:
                out["gates"][g][str(i)] = e
    cz = snap.get("gates", {}).get("cz", {})
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        e = cz.get(f"({a},{b})", cz.get(f"({b},{a})"))
        if e is not None:
            out["gates"]["cz"][f"({i},{i+1})"] = e
    fd, tmp = tempfile.mkstemp(suffix="_chainmapped.json")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f)
    return tmp
