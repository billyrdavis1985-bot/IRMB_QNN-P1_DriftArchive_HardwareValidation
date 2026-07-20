# Hybrid QNN on ibm_fez — Architecture & Experiment Design

**Target hardware:** IBM Heron r2 (`ibm_fez`, 156 qubits, heavy-hex topology, native gate set {CZ, RZ, SX, X, ID})
**Stack:** PyTorch + PennyLane (`pennylane-qiskit` plugin) + Qiskit Runtime primitives (EstimatorV2)
**Author:** Hudson Forge Technologies — QNN-P1

---

## 1. Design philosophy

The quantum component is deliberately small and purposeful. The classical network does what classical networks are good at (feature extraction, dimensionality reduction), and the variational quantum circuit (VQC) is inserted as a single trainable nonlinear layer whose feature map is classically hard to simulate at scale. Everything about the circuit design is derived backwards from ibm_fez's physical constraints rather than from textbook ansätze, because transpilation overhead (SWAP insertion, basis decomposition) is where naive QNN designs silently lose their fidelity budget.

The experiment is structured as three execution tiers so that expensive resources are only spent on questions that cheaper tiers cannot answer.

## 2. Three-tier execution model

**Tier 0 — Ideal simulation (`lightning.qubit`).** All architecture search, hyperparameter tuning, and convergence studies happen here. Exact analytic gradients via parameter-shift or adjoint differentiation. This tier answers: does the architecture learn at all, and what is its noiseless ceiling?

**Tier 1 — Calibration-seeded noisy simulation (Qiskit Aer).** A `NoiseModel` is constructed from a real ibm_fez calibration snapshot — either pulled live via `NoiseModel.from_backend()` or, more interestingly, reconstructed from your QPU Drift Collector's hourly telemetry. Because the collector gives you a *distribution* of calibration states over time, you can train against sampled noise realizations rather than a single frozen snapshot, which is a form of noise-domain randomization. This tier answers: how much of the noiseless ceiling survives realistic decoherence, gate error, and readout error, and which error-mitigation settings recover the most?

**Tier 2 — Hardware validation (`ibm_fez` via Qiskit Runtime).** Full gradient-descent training on hardware is economically irrational (see §6 cost model). Tier 2 is used for (a) forward-pass evaluation of the sim-trained model, and (b) optionally a short SPSA fine-tune (tens of steps) inside a single Runtime Session to adapt the parameters to the device's true noise channel. This tier answers: does the sim-to-hardware transfer hold, and does on-device fine-tuning close any residual gap?

The scientific deliverable is the *gap analysis across tiers*, which is a publishable result regardless of whether the QNN beats a classical baseline.

## 3. Model architecture

```
x (raw features, dim D)
   │
   ▼
Classical encoder: MLP D → 64 → n_qubits, tanh output scaled to [-π, π]
   │
   ▼
Quantum layer (n_qubits = 8, L = 3 variational blocks)
   ├── Encoding: RY(x_i) per qubit  (angle encoding — depth O(1))
   ├── Block ×L: RY(θ)·RZ(θ) per qubit → CZ linear chain (0-1, 2-3, ... then 1-2, 3-4, ...)
   └── Measurement: ⟨Z_i⟩ for each qubit  (LOCAL observables only)
   │
   ▼
Classical head: Linear n_qubits → n_classes
```

Rationale for each choice:

**Angle encoding, not amplitude encoding.** Amplitude encoding's exponential compression requires state-preparation circuits whose depth destroys the fidelity budget on NISQ hardware. Angle encoding is depth-1 and lets the classical encoder learn *what* to encode.

**CZ linear-chain entanglement.** CZ is native on Heron r2, and a linear chain maps onto a path through the heavy-hex lattice with zero SWAP insertion. A brickwork pattern of alternating even/odd pairs gives full connectivity across the register in 2 sub-layers. Ring or all-to-all entanglement would force the transpiler to insert SWAPs, roughly tripling two-qubit gate count.

**Local ⟨Z⟩ observables, not a global parity observable.** Cerezo et al. showed global cost functions induce barren plateaus even at shallow depth; local costs on shallow circuits provably avoid them. The classical head recombines the local expectations, so no expressivity is lost at the output.

**Shallow depth (L = 3) and small width (n = 8).** Barren plateau variance decays exponentially in circuit volume. Eight qubits × three blocks stays in the trainable regime, remains exactly simulable for Tier 0/1 ground truth, and keeps hardware circuit depth around 40–50 native gates after transpilation — well inside fez's coherence window.

**Identity-adjacent initialization.** Variational parameters initialize near zero so the circuit starts close to identity, which empirically avoids initializing onto a plateau.

## 4. Qubit selection on ibm_fez

Do not let the transpiler pick qubits blindly. The drift collector telemetry should drive selection of the best 8-qubit path: score each candidate linear chain in the heavy-hex coupling map by a composite of two-qubit CZ error, T1/T2, and readout error, then pass the winning chain as `initial_layout`. Because you have *hourly* snapshots, you can also check chain-quality stability over time and prefer chains that are consistently good rather than momentarily good — drift-robust layout selection is itself a small novel contribution.

## 5. Gradient strategy per tier

Tier 0 uses adjoint differentiation (fast, exact). Tier 1 uses parameter-shift with finite shots to match hardware statistics. Tier 2, if fine-tuning at all, uses SPSA: parameter-shift costs 2P circuit evaluations per step (P ≈ 8×3×2 = 48 parameters → 96 circuits/step), while SPSA costs 2 circuits per step regardless of P, at the price of stochastic gradient quality. On hardware, shot noise already randomizes gradients, so SPSA's approximation is nearly free.

## 6. Cost model (order-of-magnitude)

At 4,096 shots per circuit and ~96 circuits per parameter-shift step, one gradient step ≈ 400k shots. A modest 200-step training run ≈ 80M shots — days of queue time and a large bill. SPSA fine-tuning for 50 steps ≈ 100 circuits ≈ 400k shots total, which fits in one Session. This asymmetry is why the tier structure exists.

## 7. Error mitigation (Tier 2)

Use EstimatorV2 with `resilience_level=1` (TREX readout twirling + measurement mitigation) as the default; escalate selectively to ZNE (`resilience_level=2`) for final reported numbers only, since ZNE multiplies circuit count by the number of noise-scale factors. Enable dynamical decoupling (`XpXm` sequence) — the encoder chain leaves idle qubits during two-qubit sub-layers, and DD is nearly free insurance.

## 8. Risks and edge cases

**Barren plateaus** are mitigated by design (§3) but monitor gradient-norm statistics during Tier 0 training; if the median gradient norm collapses below ~1e-3 early, reduce L before touching anything else. **Calibration drift between Tier 1 and Tier 2** means the noise model you trained against may not be the device you evaluate on; log the calibration timestamp of the Tier 2 session and re-run Tier 1 with that exact snapshot post-hoc for a fair comparison. **Shot-noise/learning-rate coupling**: with 4,096 shots, expectation-value standard error ≈ 0.016, so learning rates below ~1e-3 make updates smaller than measurement noise — clamp accordingly. **Encoder collapse**: if the classical encoder is too powerful it can solve the task before the quantum layer, making the VQC decorative; the ablation in §9 controls for this. **API drift**: pin `pennylane`, `pennylane-qiskit`, `qiskit`, `qiskit-ibm-runtime`, `qiskit-aer` versions in the lockfile; the Runtime primitives API has broken twice in two years.

## 9. Evaluation protocol

Three models trained under identical budgets: (A) the hybrid QNN, (B) a classical control where the VQC is replaced by an MLP of matched parameter count (~48 params), and (C) the hybrid with entanglers deleted (product-state control — isolates whether entanglement contributes anything). Metrics: test accuracy per tier, Tier0→Tier1→Tier2 degradation curve, and gradient-norm trajectories. Pre-register the primary endpoint as the Tier 2 accuracy of (A) vs (B); treat (A) > (C) as the mechanistic secondary endpoint. This mirrors your MTCA/IRMB discipline: the null result is a valid, reportable outcome.

## 10. Suggested dataset

Start with a binary classification task at D ≈ 16–64 features where the encoder does honest work: `sklearn` breast-cancer (D=30) for the pipeline shakedown, then something with structure worth learning. Avoid MNIST — downsampling it to 8 qubits destroys the signal and the literature is saturated with uninformative MNIST-QNN results.
