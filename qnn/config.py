"""Experiment configuration for the ibm_fez hybrid QNN (QNN-P1).

Single source of truth for all tunables. Every tier reads from this
dataclass so that Tier 0/1/2 runs are guaranteed structurally identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import hashlib


class Tier(str, Enum):
    IDEAL = "tier0_ideal"          # lightning.qubit, analytic
    NOISY_SIM = "tier1_noisy"      # qiskit.aer + calibration noise model
    HARDWARE = "tier2_hardware"    # ibm_fez via Qiskit Runtime


@dataclass(frozen=True)
class QuantumConfig:
    n_qubits: int = 8
    n_layers: int = 3                    # variational blocks (keep shallow: barren plateaus)
    shots: int = 4096                    # ignored on Tier 0 (analytic)
    init_scale: float = 0.05             # identity-adjacent init (near-zero angles)
    # Preferred physical qubit chain on ibm_fez. Populate from drift-collector
    # chain scoring (see layout.py); None lets the transpiler choose (not recommended).
    initial_layout: tuple[int, ...] | None = (125, 124, 123, 136, 143, 142, 141, 140)


@dataclass(frozen=True)
class ClassicalConfig:
    input_dim: int = 30                  # e.g. sklearn breast-cancer
    encoder_hidden: int = 64
    n_classes: int = 2


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 5e-3                     # keep >= ~1e-3 on shot-based tiers (see DESIGN §8)
    seed: int = 8281                     # Romans 8:28 :)
    grad_norm_log_every: int = 10        # barren-plateau early warning
    spsa_steps: int = 50                 # Tier 2 fine-tune budget
    spsa_a: float = 0.05
    spsa_c: float = 0.1


@dataclass(frozen=True)
class HardwareConfig:
    backend_name: str = "ibm_fez"
    resilience_level: int = 1            # TREX/readout; use 2 (ZNE) for final numbers only
    enable_dynamical_decoupling: bool = True
    dd_sequence: str = "XpXm"
    calibration_snapshot: str | None = None  # path to drift-collector JSON for Tier 1


@dataclass(frozen=True)
class ExperimentConfig:
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def fingerprint(self) -> str:
        """SHA256 of the full config — for preregistration, IRMB-style."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"config": asdict(self), "sha256": self.fingerprint()},
                      f, indent=2, default=str)


DEFAULT = ExperimentConfig()
