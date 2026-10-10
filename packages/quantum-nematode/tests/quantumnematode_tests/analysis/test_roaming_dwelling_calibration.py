"""The roaming/dwelling calibration: a safe read of the deposit, its windows, and the held-out gate.

Covers the realworm-behavioural-validation scenario "The calibration is checked on held-out
animals": the slope is fitted on half the animals, agreement is reported on the other half, and the
gate reads kappa >= 0.6. The deposit is read without executing any class outside numpy and pandas.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from quantumnematode.validation import roaming_dwelling as rd

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))

import roaming_dwelling_calibration as cal  # noqa: E402  # pyright: ignore[reportMissingImports]


class _Malicious:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.system, ("echo pwned",))


def test_a_class_outside_numpy_and_pandas_is_refused(tmp_path: Path) -> None:
    """A pickle that would call ``os.system`` is refused before anything runs."""
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps(_Malicious()))
    with path.open("rb") as f, pytest.raises(pickle.UnpicklingError, match="refused"):
        cal._SafeUnpickler(f).load()


def _deposit(tmp_path: Path, n_animals: int = 8, n_bins: int = 40) -> Path:
    """Write a deposit-shaped pickle: alternating straight fast runs and slow turning bouts."""
    rng = np.random.default_rng(0)
    frames = n_bins * 30 + 1
    x = np.zeros((n_animals, frames))
    y = np.zeros((n_animals, frames))
    labels = np.zeros((n_animals, n_bins), dtype=bool)
    for i in range(n_animals):
        heading, px, py = 0.0, 50.0, 50.0
        for b in range(n_bins):
            roaming = (b // 10) % 2 == 0
            labels[i, b] = roaming
            for f in range(30):
                t = b * 30 + f
                if f % 15 == 0:
                    heading += 0.02 if roaming else rng.choice([-1.0, 1.0]) * 1.5
                step = 0.12 / 3 if roaming else 0.01 / 3  # mm per frame
                px += step * np.cos(heading) * 100
                py += step * np.sin(heading) * 100
                x[i, t + 1], y[i, t + 1] = px, py
        x[i, 0], y[i, 0] = 50.0, 50.0
    in_run = np.ones((n_animals, n_bins), dtype=bool)
    data = {
        "Midbody_cent_x": x[:, :-1],
        "Midbody_cent_y": y[:, :-1],
        "pixpermm": np.full((n_animals, 1), 100.0),
        "InLawnRunMask": in_run,
        "RD_states_Matrix_exog": np.ma.masked_array(labels, mask=~in_run),
    }
    path = tmp_path / "deposit.pkl"
    path.write_bytes(pickle.dumps([np.arange(0, n_bins * 30, 30), 30, data]))
    return path


def test_derive_and_calibrate_on_a_synthetic_deposit(tmp_path: Path) -> None:
    """Windows are measured from the deposit, and the calibrated line reproduces its labels."""
    windows = tmp_path / "windows.npz"
    summary = cal.derive(_deposit(tmp_path), windows)
    assert summary["animals"] == 8
    with np.load(windows) as f:
        assert f["speed"].shape == (8, 40)
        assert set(np.unique(f["labels"])) <= {rd.DWELLING, rd.ROAMING}
    result = cal.calibrate(windows)
    assert result["fit_animals"] + result["held_out_animals"] == 8
    assert result["held_out"]["kappa"] > 0.8
    assert result["gate_passes"]
    assert result["kappa_gate"] == 0.6
