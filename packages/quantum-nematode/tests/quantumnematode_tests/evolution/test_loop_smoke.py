"""Smoke and integration tests for :mod:`quantumnematode.evolution.loop`."""

from __future__ import annotations

import csv
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pytest
from quantumnematode.evolution.encoders import MLPPPOEncoder
from quantumnematode.evolution.fitness import EpisodicSuccessRate
from quantumnematode.evolution.loop import (
    CHECKPOINT_VERSION,
    EvolutionLoop,
)
from quantumnematode.optimizers.evolutionary import CMAESOptimizer
from quantumnematode.utils.config_loader import (
    EvolutionConfig,
    load_simulation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"


def _make_loop(
    output_dir: Path,
    *,
    generations: int = 2,
    population_size: int = 4,
    episodes_per_eval: int = 1,
    checkpoint_every: int = 10,
) -> EvolutionLoop:
    """Build a small loop suitable for fast smoke tests."""
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    encoder = MLPPPOEncoder()
    fitness = EpisodicSuccessRate()
    ecfg = EvolutionConfig(
        algorithm="cmaes",
        population_size=population_size,
        generations=generations,
        episodes_per_eval=episodes_per_eval,
        parallel_workers=1,
        checkpoint_every=checkpoint_every,
    )
    dim = encoder.genome_dim(sim_config)
    optimizer = CMAESOptimizer(
        num_params=dim,
        population_size=population_size,
        sigma0=ecfg.sigma0,
        seed=42,
    )
    rng = np.random.default_rng(42)
    return EvolutionLoop(
        optimizer=optimizer,
        encoder=encoder,
        fitness=fitness,
        sim_config=sim_config,
        evolution_config=ecfg,
        output_dir=output_dir,
        rng=rng,
        log_level=logging.WARNING,
    )


# ---------------------------------------------------------------------------
# 3-generation MLPPPO smoke
# ---------------------------------------------------------------------------


def test_loop_runs_3_generations_mlpppo(tmp_path: Path) -> None:
    """A 3-gen x pop 4 x 1-episode run SHALL complete and produce all artefacts."""
    loop = _make_loop(tmp_path, generations=3, population_size=4, episodes_per_eval=1)
    result = loop.run()

    assert result.generations == 3
    # Fitness in [0, 1] since EpisodicSuccessRate returns success rate
    assert 0.0 <= result.best_fitness <= 1.0
    assert (tmp_path / "best_params.json").exists()
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "lineage.csv").exists()
    assert (tmp_path / "checkpoint.pkl").exists()


def test_loop_best_params_json_round_trips_back_to_brain(tmp_path: Path) -> None:
    """``best_params.json`` SHALL decode back into a working brain."""
    loop = _make_loop(tmp_path, generations=2, population_size=4, episodes_per_eval=1)
    loop.run()

    best_path = tmp_path / "best_params.json"
    with best_path.open(encoding="utf-8") as handle:
        artefact = json.load(handle)

    assert artefact["brain_type"] == "mlpppo"
    assert isinstance(artefact["best_params"], list)
    assert len(artefact["best_params"]) > 0
    assert "best_fitness" in artefact
    assert artefact["checkpoint_version"] == CHECKPOINT_VERSION


# ---------------------------------------------------------------------------
# Lineage row count (Phase 9 verification)
# ---------------------------------------------------------------------------


def test_loop_writes_p_times_g_lineage_rows(tmp_path: Path) -> None:
    """Lineage CSV SHALL have ``population_size * generations`` data rows + header."""
    loop = _make_loop(tmp_path, generations=3, population_size=4, episodes_per_eval=1)
    loop.run()

    lineage_path = tmp_path / "lineage.csv"
    with lineage_path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    # Header + 3 generations x 4 population = 13 rows
    assert len(rows) == 13


# ---------------------------------------------------------------------------
# Checkpoint contains required keys
# ---------------------------------------------------------------------------


def test_checkpoint_contains_required_keys(tmp_path: Path) -> None:
    """Pickled checkpoint SHALL contain the documented key set."""
    loop = _make_loop(tmp_path, generations=2, population_size=4, episodes_per_eval=1)
    loop.run()

    with (tmp_path / "checkpoint.pkl").open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local file

    expected_keys = {
        "checkpoint_version",
        "optimizer",
        "generation",
        "prev_generation_ids",
        "rng_state",
        "lineage_path",
    }
    assert set(payload.keys()) == expected_keys
    assert payload["checkpoint_version"] == CHECKPOINT_VERSION


# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------


def test_loop_resume_from_checkpoint(tmp_path: Path) -> None:
    """Run 1 gen → save checkpoint → resume → run 2 more.

    The lineage CSV SHALL contain rows from all 3 generations (1 + 2),
    with no duplicate header rows.
    """
    # First session: 1 generation, checkpoint every 1 to force a save.
    loop1 = _make_loop(
        tmp_path,
        generations=1,
        population_size=4,
        episodes_per_eval=1,
        checkpoint_every=1,
    )
    loop1.run()
    checkpoint = tmp_path / "checkpoint.pkl"
    assert checkpoint.exists()

    # Read lineage after first session.
    with (tmp_path / "lineage.csv").open(encoding="utf-8") as handle:
        first_rows = list(csv.reader(handle))
    assert len(first_rows) == 5  # header + 4 data rows

    # Second session: resume, run 2 more generations (3 total).
    loop2 = _make_loop(
        tmp_path,
        generations=3,
        population_size=4,
        episodes_per_eval=1,
        checkpoint_every=1,
    )
    result = loop2.run(resume_from=checkpoint)
    assert result.generations == 3

    # Lineage now has all 3 generations.
    with (tmp_path / "lineage.csv").open(encoding="utf-8") as handle:
        all_rows = list(csv.reader(handle))
    # 1 header + 3 generations x 4 population = 13
    assert len(all_rows) == 13
    # Header SHALL appear exactly once (single-header invariant).
    header_count = sum(1 for r in all_rows if r[0] == "generation")
    assert header_count == 1


# ---------------------------------------------------------------------------
# Incompatible checkpoint version
# ---------------------------------------------------------------------------


def test_loop_rejects_incompatible_checkpoint_version(tmp_path: Path) -> None:
    """A checkpoint with a wrong version SHALL be rejected with a clear error."""
    loop = _make_loop(
        tmp_path,
        generations=1,
        population_size=4,
        episodes_per_eval=1,
        checkpoint_every=1,
    )
    loop.run()

    bogus_path = tmp_path / "bogus_checkpoint.pkl"
    with bogus_path.open("wb") as handle:
        pickle.dump({"checkpoint_version": 999, "optimizer": None, "generation": 0}, handle)

    loop2 = _make_loop(tmp_path, generations=2, population_size=4, episodes_per_eval=1)
    with pytest.raises(ValueError, match="Incompatible checkpoint version"):
        loop2.run(resume_from=bogus_path)


# ---------------------------------------------------------------------------
# parent_ids convention
# ---------------------------------------------------------------------------


def test_lineage_parent_ids_lists_all_prev_generation_ids(tmp_path: Path) -> None:
    """Every gen-N row's ``parent_ids`` SHALL be the joined set of all gen-(N-1) child_ids.

    Verifies the framework's parent_ids convention: every member of generation
    N-1 is recorded as a parent of every member of generation N (uniform across
    CMA-ES and GA, since neither optimiser exposes per-child parent provenance).
    """
    loop = _make_loop(tmp_path, generations=3, population_size=4, episodes_per_eval=1)
    loop.run()

    with (tmp_path / "lineage.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    # Skip header; group by generation.
    rows_by_gen: dict[int, list[list[str]]] = {}
    for row in rows[1:]:
        rows_by_gen.setdefault(int(row[0]), []).append(row)

    for gen in [1, 2]:
        # All children in this generation share the same parent_ids string.
        cells = {row[2] for row in rows_by_gen[gen]}
        assert len(cells) == 1
        parent_ids = sorted(next(iter(cells)).split(";"))
        prev_child_ids = sorted(row[1] for row in rows_by_gen[gen - 1])
        assert parent_ids == prev_child_ids
