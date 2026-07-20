"""Datasets for QNN-P1.

Two tasks behind one flag:

  cancer   — sklearn breast-cancer (D=30). Shakedown/validation task.
             Nearly linearly separable → expect all arms at ceiling (~96-97%).
             Its job is pipeline validation, not discrimination.

  qlabels  — quantum-generated labels. The PRIMARY, preregistrable task.
             Inputs x ~ U[-pi, pi]^8 are labeled by a FIXED random 8-qubit
             circuit (deeper than the model's ansatz: L_gen = 6 vs L_model = 3):
                 f(x) = mean_i <Z_i>  after encoding x and applying the frozen
                 random layers;  y = 1 if f(x) > median else 0
             Median thresholding guarantees exact class balance. This is the
             regime where quantum feature maps are *theorized* to help; a tie
             here is itself an informative (and publishable) result.

The generator circuit is seeded independently of the training seed so the
task is identical across all three arms and across tiers.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

GEN_SEED = 777          # task identity — never vary this within an experiment
GEN_LAYERS = 6          # deeper than the model ansatz (harder to imitate)
N_SAMPLES = 800


def make_quantum_labels(n_qubits: int, n_samples: int = N_SAMPLES,
                        gen_seed: int = GEN_SEED, gen_layers: int = GEN_LAYERS):
    """Generate (X, y) with labels from a frozen random circuit.
    Runs on lightning.qubit analytically — a few seconds, done once."""
    import pennylane as qml

    rng = np.random.default_rng(gen_seed)
    X = rng.uniform(-np.pi, np.pi, size=(n_samples, n_qubits))
    W = rng.uniform(-np.pi, np.pi, size=(gen_layers, n_qubits, 2))

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def gen_circuit(x):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        for layer in range(gen_layers):
            for i in range(n_qubits):
                qml.RY(W[layer, i, 0], wires=i)
                qml.RZ(W[layer, i, 1], wires=i)
            for i in range(0, n_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
            for i in range(1, n_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    f = np.array([np.mean(gen_circuit(x)) for x in X])
    y = (f > np.median(f)).astype(np.int64)     # exact 50/50 balance
    return X.astype(np.float32), y


def load(name: str, cfg, seed: int):
    """Returns (train_ds, test_ds, input_dim). input_dim is reported back so
    train.py can override ClassicalConfig.input_dim for the qlabels task."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if name == "cancer":
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        input_dim = X.shape[1]
        scale = True
    elif name == "qlabels":
        X, y = make_quantum_labels(cfg.quantum.n_qubits)
        input_dim = cfg.quantum.n_qubits
        scale = False          # already in [-pi, pi]; scaling would distort the task
    else:
        raise ValueError(f"unknown dataset '{name}'")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    if scale:
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    to_t = lambda a, dt: torch.tensor(np.asarray(a), dtype=dt)
    train = TensorDataset(to_t(X_tr, torch.float32), to_t(y_tr, torch.long))
    test = TensorDataset(to_t(X_te, torch.float32), to_t(y_te, torch.long))
    return train, test, input_dim
