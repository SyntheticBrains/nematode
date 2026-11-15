"""Single-thread tests on synthetic data. No test may open the real S."""

import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from ceiling import config as C  # noqa: E402
from ceiling import scorer  # noqa: E402

_real_load = scorer._load_S


@pytest.fixture(autouse=True)
def never_the_real_S(monkeypatch):
    def guarded(path, expected):
        if Path(path).resolve() == C.STAMS_PATH.resolve():
            raise AssertionError("a test tried to load the real S")
        return _real_load(path, expected)

    monkeypatch.setattr(scorer, "_load_S", guarded)
