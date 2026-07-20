"""Circuit components: angle encoding + Heron-r2-native variational ansatz.

Design constraints (see DESIGN.md §3):
  * CZ-only entanglement in a linear brickwork chain → zero SWAP insertion
    when mapped onto a heavy-hex path on ibm_fez.
  * Local <Z_i> observables only → avoids global-cost barren plateaus.
  * Depth kept shallow: post-transpilation ~40-50 native gates at n=8, L=3.
"""
from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp


def angle_encode(inputs, n_qubits: int) -> None:
    """Depth-1 angle encoding. `inputs` is expected pre-scaled to [-pi, pi]
    by the classical encoder's tanh*pi output — do NOT rescale here."""
    for i in range(n_qubits):
        qml.RY(inputs[..., i], wires=i)


def cz_brickwork(n_qubits: int) -> None:
    """Two sub-layers of CZ on a linear chain: (0,1),(2,3),... then (1,2),(3,4),...
    Full-register connectivity in depth 2, native to Heron r2."""
    for i in range(0, n_qubits - 1, 2):
        qml.CZ(wires=[i, i + 1])
    for i in range(1, n_qubits - 1, 2):
        qml.CZ(wires=[i, i + 1])


def variational_block(weights_layer, n_qubits: int) -> None:
    """One block: single-qubit RY·RZ rotations (native after basis decomposition
    to RZ/SX) followed by the CZ brickwork entangler."""
    for i in range(n_qubits):
        qml.RY(weights_layer[i, 0], wires=i)
        qml.RZ(weights_layer[i, 1], wires=i)
    cz_brickwork(n_qubits)


def make_qnode(device, n_qubits: int, n_layers: int,
               entangle: bool = True, diff_method: str = "best"):
    """Build the QNode. `entangle=False` yields the product-state control
    (model C in the DESIGN §9 ablation)."""

    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        angle_encode(inputs, n_qubits)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            if entangle:
                cz_brickwork(n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def weight_shapes(n_layers: int, n_qubits: int) -> dict:
    return {"weights": (n_layers, n_qubits, 2)}
