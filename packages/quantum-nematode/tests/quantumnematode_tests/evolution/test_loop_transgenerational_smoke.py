"""Unit + smoke tests for the per-generation transgenerational hooks.

Covers the lawn-schedule `sim_config` builder, the per-gen
`tei_prior_source` helper, and the checkpoint round-trip of the F0
substrate path.  These exercise the additions that wire the
transgenerational config block + worker-tuple `tei_prior_source` slot
through ``EvolutionLoop`` without needing to spin up real workers.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from quantumnematode.evolution.encoders import MLPPPOEncoder
from quantumnematode.evolution.fitness import EpisodicSuccessRate
from quantumnematode.evolution.inheritance import InheritanceStrategy, NoInheritance
from quantumnematode.evolution.loop import (
    EvolutionLoop,
)
from quantumnematode.evolution.transgenerational_inheritance import (
    TransgenerationalInheritance,
)
from quantumnematode.optimizers.evolutionary import CMAESOptimizer
from quantumnematode.utils.config_loader import (
    EvolutionConfig,
    LawnScheduleEntry,
    PredatorConfig,
    TransgenerationalConfig,
    load_simulation_config,
)

if TYPE_CHECKING:
    import pytest
    from quantumnematode.utils.config_loader import SimulationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"


def _sim_config_with_predators() -> SimulationConfig:
    """Load the MLPPPO foraging config and inject predator + evolution blocks.

    Keeps the brain matched to MLPPPOEncoder (so encoder.genome_dim works
    without an LSTMPPO mismatch) while exercising the lawn_schedule's
    ``environment.predators.enabled`` toggle in ``_build_per_gen_sim_config``.
    Also injects a minimal ``evolution`` block — production TEI runs
    require ``sim_config.evolution`` to be set (LearnedPerformanceFitness
    rejects None), and ``_build_per_gen_sim_config`` asserts the same
    invariant.
    """
    base = load_simulation_config(str(MLPPPO_CONFIG))
    assert base.environment is not None
    env_with_predators = base.environment.model_copy(
        update={"predators": PredatorConfig(enabled=True, count=2)},
    )
    evolution_block = EvolutionConfig(
        algorithm="cmaes",
        population_size=4,
        generations=2,
        episodes_per_eval=1,
        learn_episodes_per_eval=10,
    )
    return base.model_copy(
        update={"environment": env_with_predators, "evolution": evolution_block},
    )


def _tei_config(
    *,
    enabled: bool = True,
    decay_factor: float = 0.6,
    generations: int = 2,
) -> TransgenerationalConfig:
    """Build a TransgenerationalConfig with a complete schedule."""
    return TransgenerationalConfig(
        enabled=enabled,
        decay_factor=decay_factor,
        lawn_schedule=[
            LawnScheduleEntry(
                generation=g,
                pathogen_lawns_enabled=(g == 0),
                ppo_train_episodes=10 if g == 0 else 0,
            )
            for g in range(generations)
        ],
    )


def _make_loop(  # noqa: PLR0913 - test fixture builder; each arg is orthogonal config
    output_dir: Path,
    *,
    sim_config: SimulationConfig,
    transgenerational: TransgenerationalConfig | None = None,
    inheritance: Literal["none", "lamarckian", "baldwin", "transgenerational"] = "none",
    generations: int = 2,
    population_size: int = 4,
) -> EvolutionLoop:
    """Build a small EvolutionLoop instance suitable for helper-level tests."""
    encoder = MLPPPOEncoder()
    fitness = EpisodicSuccessRate()
    ecfg = EvolutionConfig(
        algorithm="cmaes",
        population_size=population_size,
        generations=generations,
        episodes_per_eval=1,
        # learn_episodes_per_eval > 0 is required at construction time by
        # the inheritance ↔ train-phase validator. The per-gen lawn_schedule
        # overrides it to 0 in F1+, which the train-phase bypass in
        # ``LearnedPerformanceFitness.evaluate`` permits when a substrate
        # is being inherited instead.
        learn_episodes_per_eval=10 if inheritance != "none" else 0,
        parallel_workers=1,
        checkpoint_every=10,
        inheritance=inheritance,
        transgenerational=transgenerational,
    )
    dim = encoder.genome_dim(sim_config)
    optimizer = CMAESOptimizer(
        num_params=dim,
        population_size=population_size,
        sigma0=ecfg.sigma0,
        seed=42,
    )
    rng = np.random.default_rng(42)
    strategy: InheritanceStrategy = (
        TransgenerationalInheritance() if inheritance == "transgenerational" else NoInheritance()
    )
    return EvolutionLoop(
        optimizer=optimizer,
        encoder=encoder,
        fitness=fitness,
        sim_config=sim_config,
        evolution_config=ecfg,
        output_dir=output_dir,
        rng=rng,
        inheritance=strategy,
        log_level=logging.WARNING,
    )


# ---------------------------------------------------------------------------
# _build_per_gen_sim_config
# ---------------------------------------------------------------------------


def test_build_per_gen_sim_config_returns_base_when_transgenerational_absent(
    tmp_path: Path,
) -> None:
    """When ``transgenerational`` is None the helper SHALL return the base sim_config unchanged."""
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    loop = _make_loop(tmp_path, sim_config=sim_config)
    out = loop._build_per_gen_sim_config(0)
    assert out is sim_config


def test_build_per_gen_sim_config_applies_schedule_overrides(tmp_path: Path) -> None:
    """At each generation the lawn_schedule's overrides SHALL appear in the per-gen copy."""
    sim_config = _sim_config_with_predators()
    transgenerational = _tei_config(generations=2)
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=transgenerational,
        inheritance="transgenerational",
        generations=2,
    )

    gen0 = loop._build_per_gen_sim_config(0)
    gen1 = loop._build_per_gen_sim_config(1)

    # Base config left untouched (no shared-mutation between gens).
    assert sim_config.environment is not None
    assert sim_config.environment.predators is not None
    base_enabled = sim_config.environment.predators.enabled

    # F0: pathogen lawn ON, training episodes = 10.
    assert gen0.environment is not None
    assert gen0.environment.predators is not None
    assert gen0.environment.predators.enabled is True
    assert gen0.evolution is not None
    assert gen0.evolution.learn_episodes_per_eval == 10

    # F1: pathogen lawn OFF, training episodes = 0 (inheritance-only).
    assert gen1.environment is not None
    assert gen1.environment.predators is not None
    assert gen1.environment.predators.enabled is False
    assert gen1.evolution is not None
    assert gen1.evolution.learn_episodes_per_eval == 0

    # Base config NEVER mutated — gen builders must use model_copy.
    assert sim_config.environment.predators.enabled == base_enabled


def test_build_per_gen_sim_config_returns_base_when_disabled(tmp_path: Path) -> None:
    """TEI-off arm SHALL return base sim_config regardless of the schedule.

    When ``transgenerational.enabled=False`` the helper SHALL return the
    base sim_config unchanged at every generation.
    """
    sim_config = _sim_config_with_predators()
    # The pairing validator pins ``enabled=false`` to ``inheritance=none``,
    # so this construction matches the TEI-off control arm of the
    # paired-arm ablation.
    transgenerational = _tei_config(enabled=False, generations=2)
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=transgenerational,
        inheritance="none",
        generations=2,
    )
    out_gen0 = loop._build_per_gen_sim_config(0)
    out_gen1 = loop._build_per_gen_sim_config(1)
    # Identity-passthrough — schedule overrides MUST NOT leak into the
    # control arm's per-gen sim_config.
    assert out_gen0 is sim_config
    assert out_gen1 is sim_config


# ---------------------------------------------------------------------------
# _compute_tei_prior_source
# ---------------------------------------------------------------------------


def test_compute_tei_prior_source_returns_none_at_gen_zero(tmp_path: Path) -> None:
    """At gen 0 the helper SHALL always return None (F0 has no substrate yet)."""
    sim_config = _sim_config_with_predators()
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=_tei_config(),
        inheritance="transgenerational",
    )
    # Even if a substrate path was somehow set, gen 0 must return None
    # (F0's policy IS the substrate source, not consumer).
    loop._tei_f0_substrate_path = tmp_path / "fake.tei.pt"
    assert loop._compute_tei_prior_source(0) is None


def test_compute_tei_prior_source_returns_none_when_substrate_missing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """At F1+ without an extracted substrate the helper SHALL log + return None."""
    sim_config = _sim_config_with_predators()
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=_tei_config(),
        inheritance="transgenerational",
    )
    # No substrate set → F1+ must warn and fall back to no-bias.
    with caplog.at_level(logging.WARNING, logger="quantumnematode.evolution.loop"):
        result = loop._compute_tei_prior_source(1)
    assert result is None
    assert any("substrate" in rec.message.lower() for rec in caplog.records)


def test_compute_tei_prior_source_returns_tuple_at_f1(tmp_path: Path) -> None:
    """At F1+ with a substrate path set the helper SHALL return the 3-tuple.

    The 3-tuple shape is ``(path, decay_factor, lineage_depth)`` where
    ``lineage_depth == gen``.
    """
    sim_config = _sim_config_with_predators()
    transgenerational = _tei_config(decay_factor=0.7, generations=3)
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=transgenerational,
        inheritance="transgenerational",
        generations=3,
    )
    substrate_path = tmp_path / "f0_elite.tei.pt"
    loop._tei_f0_substrate_path = substrate_path

    src_gen1 = loop._compute_tei_prior_source(1)
    src_gen2 = loop._compute_tei_prior_source(2)

    assert src_gen1 == (substrate_path, 0.7, 1)
    assert src_gen2 == (substrate_path, 0.7, 2)


def test_compute_tei_prior_source_returns_none_for_non_tei_runs(tmp_path: Path) -> None:
    """Without a transgenerational config block the helper SHALL return None at every gen."""
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    loop = _make_loop(tmp_path, sim_config=sim_config)
    for gen in range(3):
        assert loop._compute_tei_prior_source(gen) is None


def test_compute_tei_prior_source_returns_none_when_disabled(tmp_path: Path) -> None:
    """TEI-off arm SHALL return None at every gen, even with substrate set.

    When ``transgenerational.enabled=False`` the helper SHALL return
    None at every generation, even if ``_tei_f0_substrate_path`` was
    populated.
    """
    sim_config = _sim_config_with_predators()
    transgenerational = _tei_config(enabled=False, generations=3)
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=transgenerational,
        inheritance="none",
        generations=3,
    )
    # Even with a substrate path manually set, the disabled arm MUST NOT
    # propagate it to workers — the substrate is the only cross-arm
    # difference in the paired ablation.
    loop._tei_f0_substrate_path = tmp_path / "fake.tei.pt"
    for gen in range(3):
        assert loop._compute_tei_prior_source(gen) is None


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_preserves_tei_substrate_path(tmp_path: Path) -> None:
    """A populated ``_tei_f0_substrate_path`` SHALL survive a save/load cycle."""
    sim_config = _sim_config_with_predators()
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=_tei_config(),
        inheritance="transgenerational",
    )
    substrate_path = tmp_path / "elite_g0.tei.pt"
    substrate_path.write_bytes(b"stub")
    loop._tei_f0_substrate_path = substrate_path

    loop._save_checkpoint()
    with (tmp_path / "checkpoint.pkl").open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local file
    assert payload["tei_f0_substrate_path"] == str(substrate_path)

    # Fresh loop reloads the path.
    fresh = _make_loop(
        tmp_path / "fresh",
        sim_config=sim_config,
        transgenerational=_tei_config(),
        inheritance="transgenerational",
    )
    assert fresh._tei_f0_substrate_path is None
    fresh._load_checkpoint(tmp_path / "checkpoint.pkl")
    assert fresh._tei_f0_substrate_path == substrate_path


def test_checkpoint_round_trip_preserves_none_substrate_for_non_tei(tmp_path: Path) -> None:
    """A non-TEI run's checkpoint SHALL serialise tei_f0_substrate_path as None."""
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    loop = _make_loop(tmp_path, sim_config=sim_config)
    assert loop._tei_f0_substrate_path is None
    loop._save_checkpoint()
    with (tmp_path / "checkpoint.pkl").open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local file
    assert payload["tei_f0_substrate_path"] is None
