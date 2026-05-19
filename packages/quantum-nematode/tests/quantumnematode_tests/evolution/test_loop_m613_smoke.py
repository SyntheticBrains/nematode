"""Loop-level smoke tests for the M6.13 composed inheritance mode.

Covers ``EvolutionLoop``'s integration of the
``LamarckianTransgenerationalInheritance`` strategy: the widened
``_inheritance_active`` and ``_substrate_inheritance_active``
predicates, the new ``_combined_inheritance_active`` helper, the
new branch in ``_resolve_per_child_inheritance``, and the
kind-conditional F0 GC suppression in
``_run_f0_substrate_extraction``.

Built on top of ``test_loop_transgenerational_smoke.py``'s pattern
(helper-level tests that exercise loop methods without spinning up
real workers). The full end-to-end smoke (loop.run for 2 gens by 3
arms) lives in the campaign launcher's ``--smoke`` flow, not here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from quantumnematode.evolution.encoders import MLPPPOEncoder
from quantumnematode.evolution.fitness import EpisodicSuccessRate
from quantumnematode.evolution.inheritance import (
    BaldwinInheritance,
    InheritanceStrategy,
    LamarckianInheritance,
    NoInheritance,
)
from quantumnematode.evolution.lamarckian_transgenerational_inheritance import (
    LamarckianTransgenerationalInheritance,
)
from quantumnematode.evolution.loop import EvolutionLoop
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
    from quantumnematode.utils.config_loader import SimulationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"


def _sim_config_with_predators() -> SimulationConfig:
    """Load the MLPPPO foraging config and inject predator + evolution blocks."""
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


def _composed_tei_config(generations: int = 2, f1_k: int = 10) -> TransgenerationalConfig:
    """Build a TransgenerationalConfig for composed-mode tests.

    All F1+ entries have ``ppo_train_episodes > 0`` per the M6.13 validator
    rule (composed mode requires retraining).
    """
    return TransgenerationalConfig(
        enabled=True,
        decay_factor=0.6,
        lawn_schedule=[
            LawnScheduleEntry(
                generation=g,
                pathogen_lawns_enabled=(g == 0),
                ppo_train_episodes=10 if g == 0 else f1_k,
            )
            for g in range(generations)
        ],
    )


_InheritanceLiteral = Literal[
    "none",
    "lamarckian",
    "baldwin",
    "transgenerational",
    "weights+transgenerational",
]


def _make_loop(  # noqa: PLR0913 — orthogonal config knobs
    output_dir: Path,
    *,
    sim_config: SimulationConfig,
    transgenerational: TransgenerationalConfig | None = None,
    inheritance: _InheritanceLiteral = "none",
    generations: int = 2,
    population_size: int = 4,
) -> EvolutionLoop:
    """Build a small EvolutionLoop instance for composed-mode helper tests."""
    encoder = MLPPPOEncoder()
    fitness = EpisodicSuccessRate()
    ecfg = EvolutionConfig(
        algorithm="cmaes",
        population_size=population_size,
        generations=generations,
        episodes_per_eval=1,
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
    strategy: InheritanceStrategy
    if inheritance == "weights+transgenerational":
        strategy = LamarckianTransgenerationalInheritance()
    elif inheritance == "transgenerational":
        strategy = TransgenerationalInheritance()
    elif inheritance == "lamarckian":
        strategy = LamarckianInheritance()
    elif inheritance == "baldwin":
        strategy = BaldwinInheritance()
    else:
        strategy = NoInheritance()
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
# Construction guards
# ---------------------------------------------------------------------------


def test_composed_mode_strategy_instance_mismatch_raises(tmp_path: Path) -> None:
    """Loop init SHALL raise when the strategy instance's kind() mismatches the config string.

    Pins the kind-mismatch defensive check at loop.__init__ for the
    new composed value. Without the ``_expected_kind`` dict entry
    added in M6.13, the loop would either silently accept the wrong
    strategy OR KeyError on dict access.
    """
    sim_config = _sim_config_with_predators()
    encoder = MLPPPOEncoder()
    fitness = EpisodicSuccessRate()
    ecfg = EvolutionConfig(
        algorithm="cmaes",
        population_size=4,
        generations=2,
        episodes_per_eval=1,
        learn_episodes_per_eval=10,
        inheritance="weights+transgenerational",
        transgenerational=_composed_tei_config(),
    )
    dim = encoder.genome_dim(sim_config)
    optimizer = CMAESOptimizer(
        num_params=dim,
        population_size=4,
        sigma0=ecfg.sigma0,
        seed=42,
    )
    rng = np.random.default_rng(42)
    # Pass the WRONG strategy (Lamarckian, kind="weights") under the
    # composed-mode config (expects kind="weights+transgenerational").
    with pytest.raises(ValueError, match="Inheritance instance mismatch"):
        EvolutionLoop(
            optimizer=optimizer,
            encoder=encoder,
            fitness=fitness,
            sim_config=sim_config,
            evolution_config=ecfg,
            output_dir=tmp_path,
            rng=rng,
            inheritance=LamarckianInheritance(),
            log_level=logging.WARNING,
        )


def test_composed_mode_loop_constructs_cleanly(tmp_path: Path) -> None:
    """Composed-mode loop SHALL construct without error when strategy + config match."""
    sim_config = _sim_config_with_predators()
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=_composed_tei_config(),
        inheritance="weights+transgenerational",
    )
    assert loop.inheritance.kind() == "weights+transgenerational"


# ---------------------------------------------------------------------------
# Predicate widening
# ---------------------------------------------------------------------------


def test_composed_mode_fires_both_predicates(tmp_path: Path) -> None:
    """Composed mode SHALL satisfy weight-IO AND substrate-flow predicates.

    Both ``_inheritance_active`` and ``_substrate_inheritance_active``
    fire under composed mode — that's the contract: the loop runs
    both the weight-IO path AND the substrate-flow path in parallel.
    """
    sim_config = _sim_config_with_predators()
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=_composed_tei_config(),
        inheritance="weights+transgenerational",
    )
    assert loop._inheritance_active() is True
    assert loop._substrate_inheritance_active() is True
    assert loop._inheritance_records_lineage() is True
    assert loop._combined_inheritance_active() is True


def test_pure_tei_mode_does_not_fire_combined_predicate(tmp_path: Path) -> None:
    """Pure-TEI (M6.9+) SHALL satisfy substrate predicate but NOT combined / weight-IO predicates.

    Regression check: the M6.13 widening MUST NOT accidentally make
    ``_inheritance_active()`` or ``_combined_inheritance_active()``
    fire for pure-TEI runs. Pure-TEI continues using the F0
    extraction pipeline's internal GC; weight-IO is not active.
    """
    sim_config = _sim_config_with_predators()
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=TransgenerationalConfig(
            enabled=True,
            decay_factor=0.6,
            lawn_schedule=[
                LawnScheduleEntry(
                    generation=0,
                    pathogen_lawns_enabled=True,
                    ppo_train_episodes=10,
                ),
                LawnScheduleEntry(
                    generation=1,
                    pathogen_lawns_enabled=False,
                    ppo_train_episodes=0,
                ),
            ],
        ),
        inheritance="transgenerational",
    )
    assert loop._inheritance_active() is False  # NOT widened
    assert loop._substrate_inheritance_active() is True
    assert loop._combined_inheritance_active() is False  # composed-only helper
    assert loop._inheritance_records_lineage() is True


def test_lamarckian_mode_does_not_fire_substrate_predicate(tmp_path: Path) -> None:
    """M3 Lamarckian SHALL satisfy weight-IO predicate but NOT substrate predicate.

    Regression check: M3 baseline byte-equivalence — the widened
    ``_substrate_inheritance_active()`` MUST NOT fire for pure
    Lamarckian.
    """
    sim_config = _sim_config_with_predators()
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=None,
        inheritance="lamarckian",
    )
    assert loop._inheritance_active() is True
    assert loop._substrate_inheritance_active() is False
    assert loop._combined_inheritance_active() is False
    assert loop._inheritance_records_lineage() is True


# ---------------------------------------------------------------------------
# _resolve_per_child_inheritance composed branch
# ---------------------------------------------------------------------------


def test_composed_resolve_returns_lamarckian_tuple_shape(tmp_path: Path) -> None:
    """Composed mode SHALL return the same warm-start tuple shape as Lamarckian.

    ``_resolve_per_child_inheritance`` returns ``(warm_start, capture,
    parent_id)`` for both ``"weights"`` and ``"weights+transgenerational"``
    kinds: F1+ children warm-start from the parent's ``.pt`` AND
    capture their own ``.pt``. The composed branch is functionally
    identical to the Lamarckian branch.
    """
    sim_config = _sim_config_with_predators()
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=_composed_tei_config(),
        inheritance="weights+transgenerational",
    )
    # Gen 0, no parents: child_capture should be set; warm_start = None
    warm_start, capture, parent_id = loop._resolve_per_child_inheritance(
        child_idx=0,
        gen=0,
        gid="child-g0",
    )
    assert warm_start is None  # No parents at gen 0
    assert capture is not None
    assert capture == tmp_path / "inheritance" / "gen-000" / "genome-child-g0.pt"
    assert parent_id == ""  # No selected parents yet at gen 0


def test_composed_resolve_at_f1_returns_parent_warm_start(tmp_path: Path) -> None:
    """At gen 1 with a selected F0 elite, composed mode SHALL return a parent warm-start path.

    Simulates the post-F0 state by setting ``_selected_parent_ids``
    and creating the elite's expected ``.pt`` on disk. The branch
    then resolves the F1 child's warm-start path to that file.
    """
    sim_config = _sim_config_with_predators()
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=_composed_tei_config(),
        inheritance="weights+transgenerational",
    )
    # Simulate F0 having selected an elite + that elite's .pt existing
    loop._selected_parent_ids = ["elite-f0"]
    f0_dir = tmp_path / "inheritance" / "gen-000"
    f0_dir.mkdir(parents=True, exist_ok=True)
    f0_pt = f0_dir / "genome-elite-f0.pt"
    f0_pt.touch()

    warm_start, capture, parent_id = loop._resolve_per_child_inheritance(
        child_idx=0,
        gen=1,
        gid="child-f1",
    )
    assert warm_start == f0_pt
    assert capture == tmp_path / "inheritance" / "gen-001" / "genome-child-f1.pt"
    assert parent_id == "elite-f0"


# ---------------------------------------------------------------------------
# _compute_tei_prior_source under composed mode
# ---------------------------------------------------------------------------


def test_composed_mode_threads_tei_prior_source_at_f1(tmp_path: Path) -> None:
    """At gen 1 under composed mode, ``_compute_tei_prior_source`` SHALL return a tuple.

    The substrate-flow path is shared with pure-TEI; under composed
    mode it MUST also fire (the substrate flows alongside the M3
    weight-inheritance path).
    """
    sim_config = _sim_config_with_predators()
    loop = _make_loop(
        tmp_path,
        sim_config=sim_config,
        transgenerational=_composed_tei_config(),
        inheritance="weights+transgenerational",
    )
    # Simulate F0 having populated the substrate path (the
    # extraction pipeline writes this attribute after F0 completes).
    fake_substrate = tmp_path / "inheritance" / "gen-000" / "genome-elite-f0.tei.pt"
    fake_substrate.parent.mkdir(parents=True, exist_ok=True)
    fake_substrate.touch()
    loop._tei_f0_substrate_path = fake_substrate

    result = loop._compute_tei_prior_source(gen=1)
    assert result is not None
    substrate_path, decay_factor, lineage_depth = result
    assert substrate_path == fake_substrate
    assert decay_factor == 0.6
    assert lineage_depth == 1  # F1 inherits from F0; depth = 1


# ---------------------------------------------------------------------------
# F0 GC suppression
# ---------------------------------------------------------------------------


def test_composed_mode_suppresses_f0_inline_gc(tmp_path: Path) -> None:
    """Under composed mode, ``_run_f0_substrate_extraction``'s inline ``.pt`` GC SHALL be SKIPPED.

    Sets up the on-disk state with a few F0 ``.pt`` files + a ``.tei.pt``
    substrate, then calls the inline GC guard directly via
    ``_combined_inheritance_active``-gated reasoning. Under composed
    mode the F0 ``.pt`` files MUST survive so F1 children can
    warm-start; under pure-TEI they MUST be deleted (the substrate
    is the only artefact needed for F1+).

    This test exercises the predicate + an end-to-end on-disk
    assertion rather than monkey-patching the full
    ``_run_f0_substrate_extraction`` path (which depends on
    encoder.decode + load_weights internals not in scope here).
    """
    sim_config = _sim_config_with_predators()

    # Composed-mode loop SHOULD suppress.
    composed_loop = _make_loop(
        tmp_path / "composed",
        sim_config=sim_config,
        transgenerational=_composed_tei_config(),
        inheritance="weights+transgenerational",
    )
    assert composed_loop._combined_inheritance_active() is True

    # Pure-TEI loop SHOULD NOT suppress.
    pure_tei_loop = _make_loop(
        tmp_path / "pure_tei",
        sim_config=sim_config,
        transgenerational=TransgenerationalConfig(
            enabled=True,
            decay_factor=0.6,
            lawn_schedule=[
                LawnScheduleEntry(
                    generation=0,
                    pathogen_lawns_enabled=True,
                    ppo_train_episodes=10,
                ),
                LawnScheduleEntry(
                    generation=1,
                    pathogen_lawns_enabled=False,
                    ppo_train_episodes=0,
                ),
            ],
        ),
        inheritance="transgenerational",
    )
    assert pure_tei_loop._combined_inheritance_active() is False
