# Phase 1 primary-task sweep: 3 arms x 5 seeds on quantum-labels.
# Classical arm is instant; quantum arms ~1.5 min each -> ~15-20 min total.
$seeds = 8281, 1001, 2002, 3003, 4004
foreach ($s in $seeds) {
  python -m qnn.train --tier tier0_ideal --dataset qlabels --seed $s
  python -m qnn.train --tier tier0_ideal --model classical --dataset qlabels --seed $s
  python -m qnn.train --tier tier0_ideal --no-entangle --dataset qlabels --seed $s
}
python -m qnn.summarize
