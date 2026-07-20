"""Rank converted snapshots by composite error over the SELECTED chain region.

Purpose: the drift-thesis test wants calibration EXTREMES, not random days —
run Model A against the best and worst calibration states in the archive and
see whether test accuracy moves. This scores each snapshot restricted to the
qubits/edges the circuit actually uses.

Usage:
    python -m qnn.rank_snapshots exports\*_converted.json
    (PowerShell: python -m qnn.rank_snapshots (Get-ChildItem exports\*converted.json).FullName)
"""
from __future__ import annotations

import argparse
import glob
import json

from .config import DEFAULT


def score(snap: dict, chain: tuple[int, ...]) -> float | None:
    cost = 0.0
    cz = snap.get("gates", {}).get("cz", {})
    for q in chain:
        qp = snap["qubits"].get(str(q))
        if qp is None:
            return None
        cost += qp.get("readout_error", 0.01)
        cost += 1.0 / max(qp["T1_us"], 1.0)
    for a, b in zip(chain, chain[1:]):
        cost += cz.get(f"({a},{b})", cz.get(f"({b},{a})", 0.02)) * 10
    return cost


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    args = p.parse_args()

    chain = DEFAULT.quantum.initial_layout
    if chain is None:
        raise SystemExit("Set QuantumConfig.initial_layout in config.py first.")

    paths = []
    for pattern in args.paths:
        hits = glob.glob(pattern)
        paths.extend(hits if hits else [pattern])

    rows = []
    for path in sorted(paths):
        with open(path) as f:
            snap = json.load(f)
        s = score(snap, chain)
        if s is None:
            print(f"  ⚠ {path}: chain qubit missing, skipped")
            continue
        rows.append((s, snap.get("timestamp", "?"), path))

    rows.sort()
    print(f"\nchain {chain} — {len(rows)} snapshots, lower = better calibration\n")
    for s, ts, path in rows:
        print(f"  {s:.4f}  {ts}  {path}")
    if len(rows) >= 2:
        print(f"\nBEST  day: {rows[0][2]}")
        print(f"WORST day: {rows[-1][2]}")
        spread = (rows[-1][0] - rows[0][0]) / rows[0][0] * 100
        print(f"calibration spread over chain region: {spread:.1f}%")


if __name__ == "__main__":
    main()
