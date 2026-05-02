"""Smoke and integration tests for :mod:`quantumnematode.evolution.loop`."""

from __future__ import annotations

import csv
import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from quantumnematode.evolution.encoders import GenomeEncoder
    from quantumnematode.evolution.genome import Genome
    from quantumnematode.utils.config_loader import SimulationConfig

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
# Lineage row count
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
        "selected_parent_ids",
        "inheritance",
        "rng_state",
        "lineage_path",
    }
    assert set(payload.keys()) == expected_keys
    assert payload["checkpoint_version"] == CHECKPOINT_VERSION
    # Default no-inheritance run records the literal "none" + empty set.
    assert payload["inheritance"] == "none"
    assert payload["selected_parent_ids"] == []


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


# ---------------------------------------------------------------------------
# Worker initialisation policy
# ---------------------------------------------------------------------------


def test_init_worker_sets_perf_policy() -> None:
    """``_init_worker`` SHALL apply the documented perf settings.

    Locks in the contract that fitness-eval workers run with single-thread
    BLAS (no oversubscription at parallel_workers > 1) and per-step agent
    logging silenced (no f-string overhead at INFO level filter).

    The logger assertion imports the actual runtime logger used by
    ``runners.py`` and ``agent.py`` (both ``from
    quantumnematode.logging_config import logger``) and asserts
    ``isEnabledFor(INFO)`` is False — this is the actual contract the
    per-step ``isEnabledFor`` gates depend on.  Asserting on the level
    of phantom logger names like ``quantumnematode.agent.runners`` would
    pass even if the real logger were unaffected (which is exactly the
    bug we want this test to catch).
    """
    import torch
    from quantumnematode.agent.runners import logger as runtime_logger
    from quantumnematode.evolution.loop import _init_worker

    # Save and restore so the test doesn't leak state into the rest of the
    # test session.  We also force root to INFO and the runtime logger to
    # NOTSET so the only thing that can filter INFO out is _init_worker
    # setting the runtime logger's level explicitly.  Without this, the
    # global pytest WARNING level on root would mask the bug we're
    # testing for.
    original_threads = torch.get_num_threads()
    original_runtime_level = runtime_logger.level
    original_root_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.INFO)
    runtime_logger.setLevel(logging.NOTSET)
    try:
        # Sanity: pre-init, INFO is allowed (proves our setup actually
        # exposes the bug we want to catch).
        assert runtime_logger.isEnabledFor(logging.INFO)

        _init_worker(logging.INFO)

        assert torch.get_num_threads() == 1
        # The runtime logger SHALL be filtered at WARNING after init,
        # regardless of root's level.
        assert not runtime_logger.isEnabledFor(logging.INFO)
    finally:
        torch.set_num_threads(original_threads)
        runtime_logger.setLevel(original_runtime_level)
        logging.getLogger().setLevel(original_root_level)


# =============================================================================
# Hyperparameter-evolution wiring: birth_metadata + brain_type fallback
# =============================================================================


def test_loop_populates_param_schema_in_birth_metadata(tmp_path: Path) -> None:
    """Loop SHALL populate birth_metadata['param_schema'] when hyperparam_schema set.

    Records the genome instance the fitness sees and asserts the dict
    payload from build_birth_metadata is wired through.
    """
    from quantumnematode.evolution.encoders import HyperparameterEncoder
    from quantumnematode.utils.config_loader import ParamSchemaEntry

    captured_genomes: list[Genome] = []

    class _RecordingFitness:
        def evaluate(
            self,
            genome: Genome,
            sim_config: SimulationConfig,
            encoder: GenomeEncoder,
            *,
            episodes: int,
            seed: int,
        ) -> float:
            captured_genomes.append(genome)
            return 0.0

    sim_config = load_simulation_config(str(MLPPPO_CONFIG)).model_copy(
        update={
            "hyperparam_schema": [
                ParamSchemaEntry(
                    name="learning_rate",
                    type="float",
                    bounds=(1e-5, 1e-2),
                ),
            ],
        },
    )

    encoder = HyperparameterEncoder()
    ecfg = EvolutionConfig(
        algorithm="cmaes",
        population_size=2,
        generations=1,
        episodes_per_eval=1,
        parallel_workers=1,
    )
    dim = encoder.genome_dim(sim_config)
    optimizer = CMAESOptimizer(num_params=dim, population_size=2, sigma0=ecfg.sigma0, seed=42)
    loop = EvolutionLoop(
        optimizer=optimizer,
        encoder=encoder,
        fitness=_RecordingFitness(),  # type: ignore[arg-type]
        sim_config=sim_config,
        evolution_config=ecfg,
        output_dir=tmp_path,
        rng=np.random.default_rng(42),
    )
    loop.run()

    assert len(captured_genomes) > 0
    for g in captured_genomes:
        assert "param_schema" in g.birth_metadata
        # Schema travels as a list of plain dicts
        assert isinstance(g.birth_metadata["param_schema"], list)
        assert all(isinstance(e, dict) for e in g.birth_metadata["param_schema"])
        names = [e["name"] for e in g.birth_metadata["param_schema"]]
        assert "learning_rate" in names


def test_loop_birth_metadata_empty_when_no_hyperparam_schema(tmp_path: Path) -> None:
    """Without hyperparam_schema, birth_metadata SHALL be empty (weight-evolution behaviour)."""
    captured: list[Genome] = []

    class _RecordingFitness:
        def evaluate(
            self,
            genome: Genome,
            sim_config: SimulationConfig,
            encoder: GenomeEncoder,
            *,
            episodes: int,
            seed: int,
        ) -> float:
            captured.append(genome)
            return 0.0

    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    assert sim_config.hyperparam_schema is None  # weight-evolution config

    encoder = MLPPPOEncoder()
    ecfg = EvolutionConfig(
        algorithm="cmaes",
        population_size=2,
        generations=1,
        episodes_per_eval=1,
    )
    dim = encoder.genome_dim(sim_config)
    optimizer = CMAESOptimizer(num_params=dim, population_size=2, sigma0=ecfg.sigma0, seed=42)
    loop = EvolutionLoop(
        optimizer=optimizer,
        encoder=encoder,
        fitness=_RecordingFitness(),  # type: ignore[arg-type]
        sim_config=sim_config,
        evolution_config=ecfg,
        output_dir=tmp_path,
        rng=np.random.default_rng(42),
    )
    loop.run()

    assert len(captured) > 0
    # Back-compat: birth_metadata is the empty dict (since no schema)
    for g in captured:
        assert g.birth_metadata == {}


def test_lineage_brain_type_uses_sim_config_brain_name_for_hyperparam_run(
    tmp_path: Path,
) -> None:
    """Hyperparam run SHALL record sim_config.brain.name (not encoder's '') in lineage."""
    from quantumnematode.evolution.encoders import HyperparameterEncoder
    from quantumnematode.utils.config_loader import ParamSchemaEntry

    sim_config = load_simulation_config(str(MLPPPO_CONFIG)).model_copy(
        update={
            "hyperparam_schema": [
                ParamSchemaEntry(
                    name="learning_rate",
                    type="float",
                    bounds=(1e-5, 1e-2),
                ),
            ],
        },
    )

    encoder = HyperparameterEncoder()
    assert encoder.brain_name == ""  # confirms the empty-string contract

    ecfg = EvolutionConfig(
        algorithm="cmaes",
        population_size=2,
        generations=1,
        episodes_per_eval=1,
    )
    fitness = EpisodicSuccessRate()
    dim = encoder.genome_dim(sim_config)
    optimizer = CMAESOptimizer(num_params=dim, population_size=2, sigma0=ecfg.sigma0, seed=42)
    loop = EvolutionLoop(
        optimizer=optimizer,
        encoder=encoder,
        fitness=fitness,
        sim_config=sim_config,
        evolution_config=ecfg,
        output_dir=tmp_path,
        rng=np.random.default_rng(42),
    )
    loop.run()

    # Lineage CSV
    lineage_path = tmp_path / "lineage.csv"
    assert lineage_path.exists()
    with lineage_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    for row in rows:
        assert row["brain_type"] == "mlpppo", (
            f"Expected lineage row brain_type=mlpppo (from sim_config.brain.name); "
            f"got {row['brain_type']!r}"
        )

    # best_params.json
    best_params_path = tmp_path / "best_params.json"
    assert best_params_path.exists()
    artefact = json.loads(best_params_path.read_text())
    assert artefact["brain_type"] == "mlpppo"


# ---------------------------------------------------------------------------
# Lamarckian inheritance smoke
# ---------------------------------------------------------------------------


def _make_lamarckian_loop(
    output_dir: Path,
    *,
    generations: int,
    population_size: int = 4,
) -> EvolutionLoop:
    """Build a small lamarckian-inheritance loop for fast smoke tests.

    Uses HyperparameterEncoder + LearnedPerformanceFitness with K=2/L=1
    to exercise the actual inheritance code path (per-genome warm-start
    + post-train weight capture + GC) at minimum cost.
    """
    from quantumnematode.evolution.encoders import HyperparameterEncoder
    from quantumnematode.evolution.fitness import LearnedPerformanceFitness
    from quantumnematode.evolution.inheritance import LamarckianInheritance
    from quantumnematode.utils.config_loader import ParamSchemaEntry

    ecfg = EvolutionConfig(
        algorithm="cmaes",
        population_size=population_size,
        generations=generations,
        episodes_per_eval=1,
        learn_episodes_per_eval=2,
        eval_episodes_per_eval=1,
        parallel_workers=1,
        inheritance="lamarckian",
        inheritance_elite_count=1,
    )
    sim_config = load_simulation_config(str(MLPPPO_CONFIG)).model_copy(
        update={
            "hyperparam_schema": [
                ParamSchemaEntry(
                    name="learning_rate",
                    type="float",
                    bounds=(1e-5, 1e-2),
                    log_scale=True,
                ),
            ],
            "evolution": ecfg,
        },
    )
    encoder = HyperparameterEncoder()
    fitness = LearnedPerformanceFitness()
    optimizer = CMAESOptimizer(
        num_params=encoder.genome_dim(sim_config),
        population_size=population_size,
        sigma0=ecfg.sigma0,
        seed=42,
    )
    return EvolutionLoop(
        optimizer=optimizer,
        encoder=encoder,
        fitness=fitness,
        sim_config=sim_config,
        evolution_config=ecfg,
        output_dir=output_dir,
        rng=np.random.default_rng(42),
        log_level=logging.WARNING,
        inheritance=LamarckianInheritance(elite_count=1),
    )


def test_loop_with_lamarckian_inheritance_3_gens(tmp_path: Path) -> None:
    """3-gen × pop 4 lamarckian smoke: lineage + GC + checkpoint behaviour.

    Spec scenarios "Lamarckian inheritance is selectable via config and
    CLI" + "First generation runs from-scratch under any inheritance
    strategy" + "Per-genome weight checkpoints are captured and
    garbage-collected".
    """
    loop = _make_lamarckian_loop(tmp_path, generations=3, population_size=4)
    loop.run()

    # Lineage: 12 rows total, gen 0 empty inherited_from, gen 1+ shares one parent.
    with (tmp_path / "lineage.csv").open() as h:
        rows = list(csv.DictReader(h))
    assert len(rows) == 12
    by_gen: dict[int, list[dict[str, str]]] = {}
    for r in rows:
        by_gen.setdefault(int(r["generation"]), []).append(r)
    for r in by_gen[0]:
        assert r["inherited_from"] == ""
    for gen in (1, 2):
        ifrom = {r["inherited_from"] for r in by_gen[gen]}
        assert ifrom != {""}, f"gen {gen} should have non-empty inherited_from"
        assert len(ifrom) == 1, f"single elite → all gen {gen} children share one parent"

    # Inheritance dir state: only gen-002 survives (the final winner).
    inh = tmp_path / "inheritance"
    assert inh.exists()
    surviving = sorted(p.name for p in inh.iterdir())
    assert surviving == ["gen-002"]
    files = list((inh / "gen-002").glob("genome-*.pt"))
    assert len(files) == 1


def test_inheritance_directory_garbage_collection(tmp_path: Path) -> None:
    """4-gen smoke: GC SHALL remove all but the final-gen surviving parent.

    Spec scenario "Per-genome weight checkpoints are captured and
    garbage-collected" — final clause about steady-state disk usage.
    """
    loop = _make_lamarckian_loop(tmp_path, generations=4, population_size=4)
    loop.run()

    inh = tmp_path / "inheritance"
    surviving = sorted(p.name for p in inh.iterdir())
    # Earlier-generation directories were deleted (empty after GC).
    assert surviving == ["gen-003"]
    # Exactly one file in the final generation: the elite parent.
    assert len(list((inh / "gen-003").glob("genome-*.pt"))) == 1


def test_lamarckian_resume_rejects_inheritance_mismatch(tmp_path: Path) -> None:
    """Resume with a different ``inheritance`` setting SHALL raise.

    Spec scenario "Resume rejects mismatched inheritance setting".
    """
    # First session: lamarckian, save checkpoint.
    loop = _make_lamarckian_loop(tmp_path, generations=2, population_size=4)
    loop.run()
    checkpoint = tmp_path / "checkpoint.pkl"
    assert checkpoint.exists()

    # Build a fresh loop with the SAME output dir but inheritance: none —
    # i.e. the user accidentally drops --inheritance on resume.  Use the
    # default _make_loop helper which sets inheritance: none implicitly.
    loop2 = _make_loop(tmp_path, generations=4, population_size=4, episodes_per_eval=1)
    with pytest.raises(ValueError, match="inheritance setting"):
        loop2.run(resume_from=checkpoint)
