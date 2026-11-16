"""H0's and C.2's zero-shot response matrices, regenerated through M-a's settling copy.

The encoded brain is built as M-a built it (the procedure's settings, the true wiring); hinge4.json's
z is C.2 and 0.5 x 137 is H0's box centre. The primary stimulation is one unit added to neuron s's
starting state, from rest; the response is [responder, stimulated] over the 154. A parity check
against the package's batched forward (100 inputs, default_rng(9732)) must hold to 1e-5.
"""

from __future__ import annotations

import json

import numpy as np

from ceiling import config as C
from ceiling.supports import _probe

PARITY_TOL = 1e-5


def _recurrent(brain) -> np.ndarray:
    import torch

    top = brain.topology
    with torch.no_grad():
        return (top.encoder.chemical() + top.g_gap * top.encoder.gap_scale()).numpy().astype(np.float32)


def settle_from_start(R: np.ndarray, h0: np.ndarray, depth: int) -> np.ndarray:
    """M-a's spike-local copy, starting-state form: h <- tanh(h @ R), depth times (float32)."""
    h = np.asarray(h0, np.float32)
    for _ in range(depth):
        h = np.tanh(h @ R)
    return h


def _parity(brain, R: np.ndarray) -> float:
    import torch

    top = brain.topology
    rng = np.random.default_rng(C.SEED_PARITY)
    feats = np.column_stack([rng.uniform(0.05, 1.0, 100), rng.uniform(-1, 1, 100),
                             rng.uniform(-1, 1, 100), rng.uniform(0.05, 1.0, 100)]).astype(np.float32)
    with torch.no_grad():
        _, h_pkg = top.forward_with_hidden_batched(
            torch.from_numpy(feats), modulators=torch.zeros(100, brain._n_concentrations + 1))
        drive = (torch.from_numpy(feats) @ top.encoder.injection()).numpy()
    return float(np.abs(settle_from_start(R, drive, top.forward_pass_depth) - h_pkg.numpy()).max())


def responses(names154: list[str]) -> dict:
    """{"H0": matrix, "C.2": matrix, "parity": {...}} over the 154, float64."""
    from quantumnematode.connectome import encoder as enc

    probe = _probe()
    brain = probe.encoded_brain((), **probe.PROCEDURE_SETTINGS)
    top = brain.topology
    if len(top.encoder.entries) != C.N_ENCODER:
        raise ValueError("expected 137 encoder entries")
    idx = [top.neuron_names.index(n) for n in names154]
    z_c2 = [float(v) for v in json.loads(C.HINGE4_PATH.read_text())["z"]]
    out: dict = {"parity": {}}
    for name, z in (("C.2", z_c2), ("H0", [C.BOX_CENTRE] * C.N_ENCODER)):
        enc.write_encoder(brain, z)
        R = _recurrent(brain)
        parity = _parity(brain, R)
        if parity > PARITY_TOL:
            raise AssertionError(f"{name}: settling copy differs from the package by {parity}")
        E = np.zeros((len(idx), len(top.neuron_names)), np.float32)
        E[np.arange(len(idx)), idx] = 1.0
        h = settle_from_start(R, E, top.forward_pass_depth)
        out[name] = h[:, idx].T.astype(np.float64)
        out["parity"][name] = parity
    return out
