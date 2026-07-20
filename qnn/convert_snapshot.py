"""Convert QPU Drift Collector raw snapshots (Qiskit BackendProperties JSON)
into the qnn snapshot schema consumed by noise.py / preflight.py.

Collector format (observed 2026-07-15, schema_hash 17d319df8129ddab):
  { "_meta": {...},
    "properties": {
      "qubits": [ [ {name,unit,value,date}, ... ], ... ],   # index = qubit
      "gates":  [ {gate, qubits, parameters:[{name,value,...}]}, ... ]
    } }

Usage:
    python -m qnn.convert_snapshot exports_raw/*.json --outdir exports
    python -m qnn.convert_snapshot one_file.json            # writes alongside
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _qubit_params(entries: list[dict]) -> dict:
    return {e["name"]: e for e in entries}


def convert(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)

    meta = raw.get("_meta", {})
    props = raw.get("properties", raw)   # tolerate un-wrapped dumps

    out = {
        "timestamp": meta.get("calibration_ts",
                              props.get("last_update_date", "unknown")),
        "backend": meta.get("backend", props.get("backend_name", "unknown")),
        "source_file": os.path.basename(path),
        "qubits": {},
        "gates": {"sx": {}, "x": {}, "cz": {}},
    }

    for q_idx, entries in enumerate(props.get("qubits", [])):
        p = _qubit_params(entries)
        try:
            qrec = {
                "T1_us": _in_us(p["T1"]),
                "T2_us": _in_us(p["T2"]),
                "readout_error": p["readout_error"]["value"],
            }
        except KeyError as missing:
            print(f"  ⚠ qubit {q_idx}: missing {missing}, skipped", file=sys.stderr)
            continue
        out["qubits"][str(q_idx)] = qrec

    for g in props.get("gates", []):
        name = g.get("gate")
        if name not in ("sx", "x", "cz"):
            continue
        err = next((prm["value"] for prm in g.get("parameters", [])
                    if prm.get("name") == "gate_error"), None)
        if err is None:
            continue
        qs = g.get("qubits", [])
        if name == "cz" and len(qs) == 2:
            out["gates"]["cz"][f"({qs[0]},{qs[1]})"] = err
        elif len(qs) == 1:
            out["gates"][name][str(qs[0])] = err

    return out


def _in_us(entry: dict) -> float:
    v, unit = entry["value"], entry.get("unit", "us")
    if unit in ("us", "µs", ""):
        return v
    if unit == "ns":
        return v / 1000.0
    if unit == "ms":
        return v * 1000.0
    if unit == "s":
        return v * 1e6
    raise ValueError(f"unknown time unit '{unit}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="raw snapshot file(s) or globs")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    paths = []
    for pattern in args.inputs:
        hits = glob.glob(pattern)
        paths.extend(hits if hits else [pattern])

    ok = 0
    for path in paths:
        try:
            snap = convert(path)
        except (json.JSONDecodeError, OSError) as e:
            print(f"✗ {path}: {e}", file=sys.stderr)
            continue
        outdir = args.outdir or os.path.dirname(path) or "."
        os.makedirs(outdir, exist_ok=True)
        base = os.path.basename(path).replace(".json", "")
        out_path = os.path.join(outdir, f"{snap['backend']}_{base}_converted.json")
        with open(out_path, "w") as f:
            json.dump(snap, f, indent=2)
        n_cz = len(snap["gates"]["cz"])
        print(f"✓ {os.path.basename(path)} → {len(snap['qubits'])} qubits, "
              f"{n_cz} CZ edges → {out_path}")
        ok += 1
    print(f"\nconverted {ok}/{len(paths)}")


if __name__ == "__main__":
    main()
