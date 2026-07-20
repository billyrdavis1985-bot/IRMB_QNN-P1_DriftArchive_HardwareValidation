"""Hybrid model: classical encoder → VQC TorchLayer → classical head.

Also defines the two controls from DESIGN §9:
  Model A: HybridQNN(entangle=True)   — the experiment
  Model B: ClassicalControl           — VQC swapped for a param-matched MLP
  Model C: HybridQNN(entangle=False)  — product-state control
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import pennylane as qml

from .circuits import make_qnode, weight_shapes
from .config import ExperimentConfig


class Encoder(nn.Module):
    """Compresses raw features to n_qubits rotation angles in [-pi, pi]."""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        c, q = cfg.classical, cfg.quantum
        self.net = nn.Sequential(
            nn.Linear(c.input_dim, c.encoder_hidden),
            nn.ReLU(),
            nn.Linear(c.encoder_hidden, q.n_qubits),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x) * math.pi


class HybridQNN(nn.Module):
    def __init__(self, cfg: ExperimentConfig, device, entangle: bool = True,
                 diff_method: str = "best", per_sample: bool = False):
        super().__init__()
        # per_sample: route batch items through the QNode one at a time.
        # Required for shot-based parameter-shift (PennyLane #4462: no
        # gradients of broadcasted tapes). Tier 0 keeps batched adjoint.
        self.per_sample = per_sample
        q, c = cfg.quantum, cfg.classical

        self.encoder = Encoder(cfg)

        qnode = make_qnode(device, q.n_qubits, q.n_layers,
                           entangle=entangle, diff_method=diff_method)
        # TorchLayer passes an uninitialized tensor to init_method
        init = lambda t: q.init_scale * torch.randn(t.shape)  # identity-adjacent
        self.qlayer = qml.qnn.TorchLayer(
            qnode, weight_shapes(q.n_layers, q.n_qubits), init_method=init
        )
        self.head = nn.Linear(q.n_qubits, c.n_classes)

    def forward(self, x):
        angles = self.encoder(x)
        if self.per_sample:
            z = torch.stack([self.qlayer(a) for a in angles])
        else:
            z = self.qlayer(angles)      # (batch, n_qubits) of <Z_i> in [-1, 1]
        return self.head(z)

    def quantum_grad_norm(self) -> float:
        """Barren-plateau telemetry: L2 norm of the VQC parameter gradient."""
        g = next(self.qlayer.parameters()).grad
        return float(g.norm()) if g is not None else float("nan")


class ClassicalControl(nn.Module):
    """Model B: identical encoder/head, VQC replaced by an MLP whose parameter
    count matches the quantum layer (n_layers * n_qubits * 2)."""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        q = cfg.quantum
        n_q_params = q.n_layers * q.n_qubits * 2      # e.g. 48
        # Bottleneck sized so total params ≈ n_q_params
        hidden = max(1, n_q_params // (2 * q.n_qubits))
        self.encoder = Encoder(cfg)
        self.mid = nn.Sequential(
            nn.Linear(q.n_qubits, hidden), nn.Tanh(),
            nn.Linear(hidden, q.n_qubits), nn.Tanh(),
        )
        self.head = nn.Linear(q.n_qubits, cfg.classical.n_classes)

    def forward(self, x):
        return self.head(self.mid(self.encoder(x)))
