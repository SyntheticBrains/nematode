"""F.0b-i re-read on benchmark v2: the objective, the guarded plateau, the truths and the seeds.

Covers the fit-recovery spec's scenarios "The objective is the one C.1b fits", "A truth on v2 is
stable and near the wild type" and "A false plateau at wide sigma".
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import cma_recovery as cr  # noqa: E402  # pyright: ignore[reportMissingImports]
import cma_recovery_v2 as v2  # noqa: E402  # pyright: ignore[reportMissingImports]

FAMILIES = {"a": ("x", "y"), "b": ("z",)}
SCALES = {"a/x": 1.0, "a/y": 1.0, "b/z": 2.0}


def test_the_distance_weighs_families_equally_and_ignores_reported_statistics() -> None:
    """Covers "The objective is the one C.1b fits": families equal, ungraded statistics ignored."""
    truth = {"statistics": {"a/x": 0.0, "a/y": 0.0, "b/z": 0.0, "a/reported": 0.0}}
    draw = {"statistics": {"a/x": 1.0, "a/y": 1.0, "b/z": 4.0, "a/reported": 99.0}}
    # Family a: z = 1, 1 (weights 1/4 each); family b: z = 2 (weight 1/2).
    expected = ((0.25 * 1 + 0.25 * 1 + 0.5 * 4) / 1.0) ** 0.5
    assert v2.predictive_distance_v2(truth, draw, SCALES, FAMILIES) == pytest.approx(expected)


def test_undefined_statistics_count_zero_together_and_the_cap_alone() -> None:
    """A statistic undefined in both counts 0; undefined in one counts the cap."""
    both = v2.predictive_distance_v2(
        {"statistics": {"a/x": None, "a/y": 0.0, "b/z": 0.0}},
        {"statistics": {"a/x": None, "a/y": 0.0, "b/z": 0.0}},
        SCALES,
        FAMILIES,
    )
    assert both == 0.0
    one = v2.predictive_distance_v2(
        {"statistics": {"a/x": None, "a/y": 0.0, "b/z": 0.0}},
        {"statistics": {"a/x": 1.0, "a/y": 0.0, "b/z": 0.0}},
        SCALES,
        FAMILIES,
    )
    assert one == pytest.approx((0.25 * 100) ** 0.5)


def test_a_false_plateau_at_wide_sigma_does_not_stop_a_fit() -> None:
    """Covers "A false plateau at wide sigma": the guarded rule waits for sigma <= sigma0."""
    flat = [5.0] * 60
    assert v2.guarded_stop(flat, cr.SIGMA0) is True
    assert v2.guarded_stop(flat, 0.3) is False
    sigmas = [0.3] * 50 + [0.15] * 10
    assert v2.guarded_plateau_generation(flat, sigmas) == 51


def test_a_candidate_must_be_defined_and_present_on_v2() -> None:
    """Eligibility needs every statistic defined, local-search presence above 1.5, chemotaxis."""
    good = {"statistics": {"a/x": 1.0}, "presence": {"local_search": 2.0, "chemotaxis": True}}
    assert v2.eligible(good)
    assert not v2.eligible({**good, "statistics": {"a/x": None}})
    assert not v2.eligible({**good, "presence": {"local_search": 1.4, "chemotaxis": True}})
    assert not v2.eligible({**good, "presence": {"local_search": 2.0, "chemotaxis": False}})


def test_a_truth_whose_statistics_flip_is_skipped() -> None:
    """Covers "A truth on v2 is stable and near the wild type": flips and thin gaps skip it."""
    keys = ["a/x", "a/y"]
    observation = {"statistics": {"a/x": 1.0, "a/y": 1.0}}
    stable = [{"statistics": {"a/x": 1.1, "a/y": 0.9}}] * 3
    flipped = [*stable, {"statistics": {"a/x": None, "a/y": 0.9}}]
    assert v2.keep_candidate_v2(observation, stable, 0.5, 2.0, keys) == (True, "")
    keep, reason = v2.keep_candidate_v2(observation, flipped, 0.5, 2.0, keys)
    assert not keep
    assert "a/x" in reason
    keep, reason = v2.keep_candidate_v2(observation, stable, 0.5, 1.2, keys)
    assert not keep
    assert "prior - floor" in reason


def test_the_registered_seed_ranges_are_disjoint() -> None:
    """Fits, the pilot, the candidates and the discarded probe and smoke never share an animal."""
    animals = 200
    fit_max = v2.FIT_BASE + animals * (cr.FIT_BLOCK * 7 + cr.ANSWER_OFFSET + 9) + animals
    pilot_max = v2.PILOT_BASE + animals * (max(cr.PILOT_ANSWER_OFFSETS) + 9) + animals
    candidate_span = animals * (v2.CENTRE_K + 9) + animals
    ranges = [
        (v2.FIT_BASE, fit_max),
        (v2.PILOT_BASE, pilot_max),
        (v2.PILOT_A1_BASE, v2.PILOT_A1_BASE + (pilot_max - v2.PILOT_BASE)),
        (v2.OBJECTIVE_PROBE_BASE, v2.OBJECTIVE_PROBE_BASE + animals * (cr.ANSWER_OFFSET + 10)),
        (v2.CANDIDATE_BASE, v2.CANDIDATE_BASE + candidate_span),
        (v2.PROBE_BASE, v2.PROBE_BASE + candidate_span),
        (v2.SMOKE_BASE, v2.SMOKE_BASE + animals * 32),
    ]
    ranges.sort()
    for (_, end), (start, _) in itertools.pairwise(ranges):
        assert end <= start
    assert v2.REPEAT_K + cr.REPEATS * 43 <= v2.REFERENCE_K
    assert v2.REFERENCE_K + v2.REFERENCES <= v2.CENTRE_K


def test_the_uncapped_objective_keeps_a_far_statistic_s_distance() -> None:
    """Uncapped, a statistic 30 scales off still ranks above one 20 off; capped, both read 10."""
    truth = {"statistics": {"a/x": 0.0, "a/y": 0.0, "b/z": 0.0}}
    far = {"statistics": {"a/x": 30.0, "a/y": 0.0, "b/z": 0.0}}
    nearer = {"statistics": {"a/x": 20.0, "a/y": 0.0, "b/z": 0.0}}
    capped = [v2.predictive_distance_v2(truth, d, SCALES, FAMILIES) for d in (far, nearer)]
    assert capped[0] == capped[1]
    uncapped = [
        v2.predictive_distance_v2(truth, d, SCALES, FAMILIES, capped=False) for d in (far, nearer)
    ]
    assert uncapped[0] > uncapped[1]
    undefined = {"statistics": {"a/x": None, "a/y": 0.0, "b/z": 0.0}}
    assert v2.predictive_distance_v2(truth, undefined, SCALES, FAMILIES, capped=False) == capped[1]


def test_a_torn_last_line_is_ignored_and_rerun(tmp_path: Path) -> None:
    """A store's unterminated last line is not read, so its job stays eligible to run."""
    path = tmp_path / "store.jsonl"
    path.write_text('{"k": 1, "z": [0.5]}\n{"k": 2, "z": [0.')
    assert list(v2.read_records(path)) == [1]
    v2._drop_torn_line(path)
    assert path.read_text() == '{"k": 1, "z": [0.5]}\n'


def test_a_stored_record_is_reused_only_for_its_own_inputs(tmp_path: Path) -> None:
    """A stored k with other parameters, or another seed base, is refused rather than reused."""
    path = tmp_path / "store.jsonl"
    path.write_text('{"k": 1, "z": [0.5], "seed_base": 100}\n')
    assert v2.run_jobs([(1, [0.5])], 100, path, workers=1)[1]["z"] == [0.5]
    with pytest.raises(ValueError, match="other parameters"):
        v2.run_jobs([(1, [0.6])], 100, path, workers=1)
    with pytest.raises(ValueError, match="seed base"):
        v2.run_jobs([(1, [0.5])], 200, path, workers=1)


def test_a_fit_saved_with_other_callbacks_is_refused(tmp_path: Path) -> None:
    """A saved fit is neither returned nor resumed under another simulator or stop rule."""
    import json

    saved = {
        "fit": 0,
        "simulate": cr._identity(v2.simulate_v2),
        "stop_rule": cr._identity(v2.guarded_stop),
        "generation": 1,
        "medians": [1.0],
        "sigmas": [0.2],
        "answers": [[0.5] * 20],
        "seconds": [1.0],
        "stop": {"generation": 1, "by": "cap"},
    }
    (tmp_path / "fit0.json").write_text(json.dumps(saved))
    common = {
        "fit": 0,
        "cma_seed": 1,
        "seed_base": 0,
        "cap": 1,
        "use_rule": True,
        "state_dir": tmp_path,
        "pool": None,
        "scales": {},
    }
    same = cr.run_fit({}, **common, simulate=v2.simulate_v2, stop=v2.guarded_stop)
    assert same["stop"] == saved["stop"]
    with pytest.raises(RuntimeError, match="stop_rule"):
        cr.run_fit({}, **common, simulate=v2.simulate_v2)
    with pytest.raises(RuntimeError, match="simulate"):
        cr.run_fit({}, **common, stop=v2.guarded_stop)
