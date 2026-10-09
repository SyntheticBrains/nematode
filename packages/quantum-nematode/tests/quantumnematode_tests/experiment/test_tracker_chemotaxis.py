"""The tracker records the simulated chemotaxis index and its band, and no literature verdict.

Covers the experiment-tracking requirement "No literature verdict on the simulated chemotaxis index"
(a tracked run records no literature verdict; older records still load).
"""

from __future__ import annotations

from quantumnematode.experiment.metadata import ResultsMetadata
from quantumnematode.experiment.tracker import aggregate_results_metadata
from quantumnematode.report.dtypes import SimulationResult, TerminationReason
from quantumnematode.validation.chemotaxis import ChemotaxisMetrics

_RETIRED = ("biological_ci_range", "biological_ci_typical", "matches_biology", "literature_source")


def _results(n: int = 40) -> tuple[list[SimulationResult], list[tuple[int, ChemotaxisMetrics]]]:
    results, metrics = [], []
    for run in range(1, n + 1):
        results.append(
            SimulationResult(
                run=run,
                steps=50,
                path=[(0, 0)],
                total_reward=1.0,
                last_total_reward=1.0,
                termination_reason=TerminationReason.GOAL_REACHED,
                success=True,
            ),
        )
        metrics.append(
            (
                run,
                ChemotaxisMetrics(
                    chemotaxis_index=0.8,
                    time_in_attractant=0.9,
                    approach_frequency=0.7,
                    path_efficiency=0.6,
                    total_steps=50,
                    steps_in_attractant=45,
                    steps_in_control=5,
                ),
            ),
        )
    return results, metrics


def test_a_tracked_run_records_no_literature_verdict() -> None:
    """The index and its band are recorded; the four retired literature fields are None."""
    results, metrics = _results()
    meta = aggregate_results_metadata(results, precomputed_chemotaxis=metrics)
    assert meta.post_convergence_chemotaxis_index is not None
    assert meta.chemotaxis_validation_level == "excellent"
    for field in _RETIRED:
        assert getattr(meta, field) is None


def test_metrics_derived_from_results_record_no_literature_verdict() -> None:
    """Without precomputed metrics the index comes from each run's path and food; no verdict."""
    results = [
        SimulationResult(
            run=run,
            steps=50,
            path=[(10, 10)] * 50,
            food_history=[[(10, 10)]] * 50,
            total_reward=1.0,
            last_total_reward=1.0,
            termination_reason=TerminationReason.GOAL_REACHED,
            success=True,
        )
        for run in range(1, 41)
    ]
    meta = aggregate_results_metadata(results)
    assert meta.post_convergence_chemotaxis_index == 1.0
    assert meta.chemotaxis_validation_level == "excellent"
    for field in _RETIRED:
        assert getattr(meta, field) is None


def test_older_records_still_load() -> None:
    """A record written with the literature fields set keeps its values."""
    meta = ResultsMetadata.model_validate(
        {
            "total_runs": 1,
            "success_rate": 1.0,
            "avg_steps": 50.0,
            "avg_reward": 1.0,
            "biological_ci_range": [0.5, 0.85],
            "biological_ci_typical": 0.7,
            "matches_biology": True,
            "literature_source": "Bargmann et al. (1993). Cell 74(3):515-527",
        },
    )
    assert meta.matches_biology is True
    assert meta.biological_ci_range == (0.5, 0.85)
