"""Preflight checks — run this BEFORE any Tier 1/2 execution. Free to run.

Three checks:
  1. snapshot   — validate a drift-collector JSON against the expected schema
  2. transpile  — compile the ansatz against the ibm_fez target and report
                  depth / 2q-gate count / SWAP insertion (should be ZERO SWAPs)
  3. chains     — score 8-qubit linear chains across snapshots and print the
                  recommended initial_layout for config.py

Usage:
    python -m qnn.preflight snapshot  path/to/snap.json
    python -m qnn.preflight transpile [--layout 4,5,6,7,8,9,10,11]
    python -m qnn.preflight chains    snap1.json snap2.json ...
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import DEFAULT

REQUIRED_QUBIT_FIELDS = {"T1_us", "T2_us", "readout_error"}


def check_snapshot(path: str) -> bool:
    with open(path) as f:
        snap = json.load(f)
    ok = True
    if "qubits" not in snap:
        print("✗ missing top-level 'qubits'"); return False
    for q, props in snap["qubits"].items():
        missing = REQUIRED_QUBIT_FIELDS - set(props)
        if missing:
            print(f"✗ qubit {q}: missing {missing}"); ok = False
        elif props["T2_us"] > 2 * props["T1_us"]:
            print(f"⚠ qubit {q}: T2 > 2*T1 (will be clamped)")
    cz = snap.get("gates", {}).get("cz", {})
    if not cz:
        print("⚠ no CZ errors in snapshot — 2q noise will be omitted"); 
    print(f"{'✓' if ok else '✗'} {path}: {len(snap['qubits'])} qubits, {len(cz)} CZ edges")
    return ok


def build_reference_circuit(n_qubits: int, n_layers: int):
    """The ansatz as a bare Qiskit circuit for transpilation analysis."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(0.1, i)                      # encoding placeholder
    for _ in range(n_layers):
        for i in range(n_qubits):
            qc.ry(0.1, i); qc.rz(0.1, i)
        for i in range(0, n_qubits - 1, 2):
            qc.cz(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cz(i, i + 1)
    qc.measure_all()
    return qc


def check_transpile(layout: list[int] | None) -> None:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService

    cfg = DEFAULT
    service = QiskitRuntimeService()
    backend = service.backend(cfg.hardware.backend_name)
    qc = build_reference_circuit(cfg.quantum.n_qubits, cfg.quantum.n_layers)

    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, initial_layout=layout)
    isa = pm.run(qc)

    ops = isa.count_ops()
    n_2q = ops.get("cz", 0)
    n_swap_equiv = max(0, n_2q - expected_cz(cfg))
    print(f"backend           : {backend.name}")
    print(f"logical→physical  : {layout or 'transpiler-chosen'}")
    print(f"depth             : {isa.depth()}")
    print(f"op counts         : {dict(ops)}")
    print(f"CZ count          : {n_2q} (expected {expected_cz(cfg)})")
    if n_swap_equiv:
        print(f"✗ ~{n_swap_equiv // 3} SWAPs inserted — layout is NOT a valid "
              f"path in the coupling map. Re-run chain scoring.")
        sys.exit(1)
    print("✓ zero SWAP insertion — layout maps cleanly onto heavy-hex path")


def expected_cz(cfg) -> int:
    n, L = cfg.quantum.n_qubits, cfg.quantum.n_layers
    return L * ((n // 2) + ((n - 1) // 2))


def run_chain_scoring(snapshot_paths: list[str]) -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from .noise import score_linear_chains

    cfg = DEFAULT
    service = QiskitRuntimeService()
    backend = service.backend(cfg.hardware.backend_name)
    edges = [tuple(e) for e in backend.coupling_map.get_edges()]

    scored = score_linear_chains(snapshot_paths, edges,
                                 chain_len=cfg.quantum.n_qubits)
    print(f"scored {len(scored)} candidate chains over {len(snapshot_paths)} snapshots")
    for cost, chain in scored[:5]:
        print(f"  cost {cost:.4f}  chain {chain}")
    best = scored[0][1]
    print(f"\n→ set in config.py:  initial_layout = {best}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("snapshot"); s1.add_argument("path")
    s2 = sub.add_parser("transpile")
    s2.add_argument("--layout", type=str, default=None,
                    help="comma-separated physical qubits, e.g. 4,5,6,7,8,9,10,11")
    s3 = sub.add_parser("chains"); s3.add_argument("paths", nargs="+")
    a = p.parse_args()

    if a.cmd == "snapshot":
        sys.exit(0 if check_snapshot(a.path) else 1)
    elif a.cmd == "transpile":
        layout = [int(x) for x in a.layout.split(",")] if a.layout else None
        check_transpile(layout)
    elif a.cmd == "chains":
        run_chain_scoring(a.paths)
