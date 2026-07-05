"""Agreement grading for behavioural klinotaxis bias statistics vs literature references.

Reduces a per-seed set of bias-statistic values (down/up turn-rate ratio; weathervane slope) to
a mean + an 80% bootstrap CI, then grades it against a :class:`BiasCurveReference` (§3) as
REPRODUCED / PARTIAL / ABSENT.

The grading is deliberately conservative and behaviour-level (see the reference notes): a ranged
reference (e.g. the klinokinesis ratio) grades REPRODUCED only when a significant correct-sign bias
*and* a literature-range-overlapping magnitude are both present; a sign-only reference (e.g. the
weathervane slope) grades on direction alone. Verdicts:

- **REPRODUCED**: the 80% CI excludes the no-bias null on the reference's sign side (a significant
  correct-direction bias), and — for a ranged reference — the CI overlaps the literature range.
- **PARTIAL**: the mean leans the correct way but the CI includes the null (direction not
  significant), or a significant bias whose magnitude falls outside the literature range.
- **ABSENT**: the mean does not lean the reference's direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .datasets import BiasCurveReference

_BOOTSTRAP_RESAMPLES = 1000
_CI_LEVEL = 0.80
_BOOTSTRAP_SEED = 42


class Verdict(StrEnum):
    """Per-curve agreement verdict against a behaviour-level literature reference."""

    REPRODUCED = "REPRODUCED"
    PARTIAL = "PARTIAL"
    ABSENT = "ABSENT"


@dataclass(slots=True)
class AgreementResult:
    """A graded bias statistic: the reduced value + 80% bootstrap CI + verdict vs the reference."""

    statistic: str
    verdict: Verdict
    mean: float
    ci_lo: float
    ci_hi: float
    n: int
    null_value: float
    sign: int
    magnitude_range: tuple[float, float] | None
    citation: str

    def to_dict(self) -> dict:
        """JSON-serialisable summary (verdict as its string value)."""
        return {
            "statistic": self.statistic,
            "verdict": self.verdict.value,
            "mean": self.mean,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "n": self.n,
            "null_value": self.null_value,
            "sign": self.sign,
            "magnitude_range": list(self.magnitude_range) if self.magnitude_range else None,
            "citation": self.citation,
        }


def bootstrap_ci(values: Sequence[float]) -> tuple[float, float, float]:
    """Mean + 80% bootstrap CI (mean, ci_lo, ci_hi) over per-seed values; seeded (rng=42)."""
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    boots = np.array(
        [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(_BOOTSTRAP_RESAMPLES)],
    )
    alpha = 1.0 - _CI_LEVEL
    return mean, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1.0 - alpha / 2))


def _leans_correct(mean: float, null_value: float, sign: int) -> bool:
    return (mean - null_value) * sign > 0.0


def _ci_excludes_null(ci_lo: float, ci_hi: float, null_value: float, sign: int) -> bool:
    """Return whether the whole CI sits on the reference's sign side of the null (significant)."""
    return ci_lo > null_value if sign > 0 else ci_hi < null_value


def _ranges_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return lo_a <= hi_b and lo_b <= hi_a


def grade_statistic(
    values: Sequence[float],
    reference: BiasCurveReference,
) -> AgreementResult:
    """Grade per-seed statistic values vs a reference (REPRODUCED/PARTIAL/ABSENT)."""
    finite = [float(v) for v in values if np.isfinite(v)]
    mean, ci_lo, ci_hi = bootstrap_ci(finite)
    null, sign = reference.null_value, reference.sign

    if not finite or not _leans_correct(mean, null, sign):
        verdict = Verdict.ABSENT
    elif not _ci_excludes_null(ci_lo, ci_hi, null, sign):
        verdict = Verdict.PARTIAL  # leans correct but the CI includes the no-bias null
    elif reference.magnitude_range is None:
        verdict = Verdict.REPRODUCED  # sign-only reference: a significant correct-direction bias
    else:
        lo, hi = reference.magnitude_range
        in_range = _ranges_overlap(ci_lo, ci_hi, lo, hi) or (lo <= mean <= hi)
        verdict = Verdict.REPRODUCED if in_range else Verdict.PARTIAL

    return AgreementResult(
        statistic=reference.statistic,
        verdict=verdict,
        mean=mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n=len(finite),
        null_value=null,
        sign=sign,
        magnitude_range=reference.magnitude_range,
        citation=reference.citation,
    )
