"""Aggregate results: mean/std of test-acc-at-best-val per arm and snapshot."""
import glob, json, statistics as st, re
from collections import defaultdict

groups = defaultdict(list)
for path in glob.glob("results_*.json") + glob.glob("*/results_*.json"):
    with open(path) as f:
        r = json.load(f)
    if "test_acc_at_best_val" not in r:
        continue
    arm = f'{r["model"]}{"" if r.get("entangle", True) else "_noent"}'
    snap = r.get("snapshot") or "-"
    if snap != "-":
        m = re.search(r"\d{4}-\d{2}-\d{2}", snap)
        snap = m.group(0) if m else snap[:16]
    groups[(r["tier"], r["dataset"], arm, snap)].append(r["test_acc_at_best_val"])

print(f'{"tier":<14}{"dataset":<9}{"arm":<15}{"snapshot":<10}{"n":>3}  {"mean":>7}  {"std":>6}')
for (tier, ds, arm, snap), accs in sorted(groups.items()):
    mean = st.mean(accs)
    std = st.stdev(accs) if len(accs) > 1 else 0.0
    print(f"{tier:<14}{ds:<9}{arm:<15}{snap:<10}{len(accs):>3}  {mean:>7.4f}  {std:>6.4f}")
