"""Inputs: T (the training matrix only), the 154 names, and the 302-neuron Emmons 2024 connectome.

T is decoded from its own zip member; the test member is never opened here. The file's SHA-256 is
checked over its bytes before T is decoded.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from ceiling import config as C


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_T(path: Path = C.STAMS_PATH, expected_sha256: str = C.STAMS_SHA256) -> np.ndarray:
    """The training matrix [responder, stimulated], float64, with its diagonal masked to NaN."""
    found = file_sha256(path)
    if found != expected_sha256:
        raise ValueError(f"{path} has SHA-256 {found}, expected {expected_sha256}")
    with zipfile.ZipFile(path) as archive:
        T = np.lib.format.read_array(io.BytesIO(archive.read(C.TRAIN_MEMBER)))
    T = np.array(T, dtype=np.float64)
    if T.shape != (C.N_MEASURED, C.N_MEASURED):
        raise ValueError(f"T has shape {T.shape}")
    np.fill_diagonal(T, np.nan)
    return T


def load_names(path: Path = C.NAMES_PATH) -> list[str]:
    names = json.loads(Path(path).read_text())
    if len(names) != C.N_MEASURED or len(set(names)) != C.N_MEASURED:
        raise ValueError("expected 154 distinct names")
    return names


@functools.cache
def connectome():
    """The package's Emmons 2024 hermaphrodite connectome."""
    from quantumnematode.connectome.loader import load_emmons_2024_hermaphrodite

    return load_emmons_2024_hermaphrodite()


def connectome_names(conn=None) -> list[str]:
    """The 302 names in the brain's index order (sorted, as ConnectomeTopology orders them)."""
    names = sorted((conn or connectome()).neurons)
    if len(names) != C.N_CONNECTOME:
        raise ValueError(f"expected 302 neurons, found {len(names)}")
    return names


def measured_index(names154: list[str], names302: list[str]) -> np.ndarray:
    """Position of each of the 154 in the 302, by name."""
    position = {name: k for k, name in enumerate(names302)}
    missing = [n for n in names154 if n not in position]
    if missing:
        raise ValueError(f"names absent from the connectome: {missing}")
    return np.array([position[n] for n in names154], dtype=np.int64)
