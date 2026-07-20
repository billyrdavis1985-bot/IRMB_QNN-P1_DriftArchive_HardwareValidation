"""Phase 5: hardware evaluation on ibm_fez. Evaluation only - no training.

Protocol: probe first (--limit 4) to verify broadcast + device path on a
tiny job, THEN full run. Per PREREGISTRATION.md.
"""
from __future__ import annotations
import argparse, dataclasses, json, time
import torch
from torch.utils.data import DataLoader
from .config import DEFAULT, Tier
from .devices import make_device
from .model import HybridQNN
from .train import load_data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--no-entangle", action="store_true")
    p.add_argument("--seed", type=int, default=8281)
    p.add_argument("--dataset", choices=["cancer", "qlabels"], default="qlabels")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N test samples (probe: 4)")
    a = p.parse_args()

    cfg = dataclasses.replace(
        DEFAULT,
        train=dataclasses.replace(DEFAULT.train, seed=a.seed),
        quantum=dataclasses.replace(DEFAULT.quantum, shots=a.shots),
    )
    _, test_ds, cfg = load_data(cfg, a.dataset)
    if a.limit:
        test_ds = torch.utils.data.Subset(test_ds, range(a.limit))
    test_dl = DataLoader(test_ds, batch_size=a.batch)

    device = make_device(Tier.HARDWARE, cfg)
    model = HybridQNN(cfg, device, entangle=not a.no_entangle,
                      diff_method="parameter-shift", per_sample=False)
    model.load_state_dict(torch.load(a.weights))
    model.eval()

    stamp = {"wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "backend": cfg.hardware.backend_name, "shots": a.shots,
             "weights": a.weights, "entangle": not a.no_entangle,
             "seed": a.seed, "limit": a.limit}
    try:
        props = device.backend.properties()
        stamp["calibration_time"] = str(getattr(props, "last_update_date", "n/a"))
    except Exception as e:
        stamp["calibration_time"] = f"unavailable ({e})"
    print("calibration stamp:", stamp["calibration_time"], flush=True)

    correct = total = 0
    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_dl):
            t0 = time.time()
            pred = model(xb).argmax(dim=1)
            correct += int((pred == yb).sum())
            total += len(yb)
            print(f"  job-batch {i}: {len(yb)} samples, "
                  f"running acc {correct/total:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
    acc = correct / total
    stamp["test_acc_hardware"] = acc
    tag = "probe" if a.limit else "full"
    out = a.weights.replace(".pt", f"_hw_{tag}.json").replace("best-day\\", "")
    with open(out, "w") as f:
        json.dump(stamp, f, indent=2)
    print(f"\nHARDWARE accuracy ({total} samples, {a.shots} shots): {acc:.4f}  ->  {out}")

if __name__ == "__main__":
    main()
