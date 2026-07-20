"""Device factory: one function, three tiers.

Tier 0: lightning.qubit          — analytic, adjoint gradients
Tier 1: qiskit.aer + NoiseModel  — calibration-seeded noisy simulation
Tier 2: qiskit.remote (ibm_fez)  — real hardware via Qiskit Runtime

Version pinning matters here (DESIGN §8 'API drift'). Verified against:
  pennylane>=0.38, pennylane-qiskit>=0.38, qiskit>=1.2,
  qiskit-aer>=0.15, qiskit-ibm-runtime>=0.30
Re-verify primitive/options signatures against current docs before a paid run.
"""
from __future__ import annotations

import pennylane as qml

from .config import ExperimentConfig, Tier
from .noise import build_noise_model


def make_device(tier: Tier, cfg: ExperimentConfig):
    n = cfg.quantum.n_qubits

    if tier is Tier.IDEAL:
        # Analytic simulation; pair with diff_method="adjoint" in the QNode.
        return qml.device("lightning.qubit", wires=n)

    if tier is Tier.NOISY_SIM:
        noise_model = build_noise_model(cfg)
        return qml.device(
            "qiskit.aer",
            wires=n,
            shots=cfg.quantum.shots,
            noise_model=noise_model,
        )

    if tier is Tier.HARDWARE:
        # Imported lazily so Tiers 0/1 don't require runtime credentials.
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()  # uses saved account
        backend = service.backend(cfg.hardware.backend_name)

        kwargs = {}
        if cfg.quantum.initial_layout is not None:
            # Forwarded to the plugin's transpile step; verify against your
            # installed pennylane-qiskit version (see preflight transpile).
            kwargs["initial_layout"] = list(cfg.quantum.initial_layout)
        dev = qml.device(
            "qiskit.remote",
            wires=n,
            backend=backend,
            shots=cfg.quantum.shots,
            **kwargs,
        )
        _apply_runtime_options(dev, cfg)
        return dev

    raise ValueError(f"Unknown tier: {tier}")


def _apply_runtime_options(dev, cfg: ExperimentConfig) -> None:
    """Best-effort application of Estimator options (resilience, DD).
    The pennylane-qiskit device exposes the underlying primitive options;
    the exact attribute path has shifted across releases, hence the guard."""
    try:
        opts = dev._estimator_options  # noqa: SLF001 (plugin-internal)
        opts.resilience_level = cfg.hardware.resilience_level
        if cfg.hardware.enable_dynamical_decoupling:
            opts.dynamical_decoupling.enable = True
            opts.dynamical_decoupling.sequence_type = cfg.hardware.dd_sequence
    except AttributeError:
        import warnings
        warnings.warn(
            "Could not set runtime options via plugin internals; "
            "configure resilience/DD when constructing the Estimator "
            "session manually (see eval_hardware.py fallback path)."
        )
