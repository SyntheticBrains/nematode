"""Tests for the F0 Substrate Extraction Pipeline integration.

Covers the loop-level pieces of the transgenerational inheritance
pipeline: ``_substrate_inheritance_active()`` helper, the F0
``weight_capture_path`` enable in ``_resolve_per_child_inheritance``,
and the on-disk state at each pipeline step (capture → load →
extract → save → GC).

The dataclass-level extraction determinism and signature tests live
in ``test_transgenerational_memory.py``; the strategy-level
Protocol-conformance tests live in
``test_transgenerational_inheritance.py``; this module focuses on
how the loop wires them together.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from quantumnematode.evolution.encoders import MLPPPOEncoder
from quantumnematode.evolution.fitness import EpisodicSuccessRate

if TYPE_CHECKING:
    import pytest
from quantumnematode.evolution.inheritance import (
    BaldwinInheritance,
    LamarckianInheritance,
    NoInheritance,
)
from quantumnematode.evolution.loop import EvolutionLoop
from quantumnematode.evolution.transgenerational_inheritance import (
    TransgenerationalInheritance,
)
from quantumnematode.optimizers.evolutionary import CMAESOptimizer
from quantumnematode.utils.config_loader import (
    EvolutionConfig,
    load_simulation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"


def _make_baseline_loop(output_dir: Path) -> EvolutionLoop:
    """Build a minimal EvolutionLoop with inheritance=none.

    Tests then monkey-patch ``loop.inheritance`` to a
    ``TransgenerationalInheritance()`` instance after construction —
    the kind()-vs-config validator runs only at construction, so
    post-construction substitution is safe for unit-testing the loop
    helpers. This is cheaper than constructing a full TEI-configured
    loop (which requires hyperparam_schema + non-zero
    learn_episodes_per_eval + matching encoder).
    """
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    encoder = MLPPPOEncoder()
    fitness = EpisodicSuccessRate()
    ecfg = EvolutionConfig(
        algorithm="cmaes",
        population_size=4,
        generations=2,
        episodes_per_eval=1,
        parallel_workers=1,
        checkpoint_every=10,
    )
    dim = encoder.genome_dim(sim_config)
    optimizer = CMAESOptimizer(
        num_params=dim,
        population_size=4,
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
# _substrate_inheritance_active() helper
# ---------------------------------------------------------------------------


def test_substrate_inheritance_active_returns_true_for_transgenerational(tmp_path: Path) -> None:
    """``_substrate_inheritance_active()`` SHALL return True for TransgenerationalInheritance."""
    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = TransgenerationalInheritance()
    assert loop._substrate_inheritance_active() is True


def test_substrate_inheritance_active_returns_false_for_other_strategies(tmp_path: Path) -> None:
    """``_substrate_inheritance_active()`` SHALL return False for non-TEI strategies."""
    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = NoInheritance()
    assert loop._substrate_inheritance_active() is False
    loop.inheritance = LamarckianInheritance()
    assert loop._substrate_inheritance_active() is False
    loop.inheritance = BaldwinInheritance()
    assert loop._substrate_inheritance_active() is False


# ---------------------------------------------------------------------------
# _resolve_per_child_inheritance transgenerational branch
# ---------------------------------------------------------------------------


def test_resolve_per_child_inheritance_transgenerational_f0_captures_weights(
    tmp_path: Path,
) -> None:
    """Under transgenerational at gen 0, ``_resolve_per_child_inheritance`` SHALL set capture_path.

    The capture path is non-None (an ``.pt`` file under
    ``inheritance/gen-000/``) so the F0 worker writes trained weights
    to disk for the post-eval Substrate Extraction Pipeline.
    """
    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = TransgenerationalInheritance()
    warm_start, capture, inherited_from = loop._resolve_per_child_inheritance(
        child_idx=0,
        gen=0,
        gid="gid-a",
    )
    assert warm_start is None
    assert capture is not None
    assert capture.name == "genome-gid-a.pt"
    assert capture.parent.name == "gen-000"
    assert inherited_from == ""  # No parent at gen 0.


def test_resolve_per_child_inheritance_transgenerational_f1_no_capture(tmp_path: Path) -> None:
    """At gen 1+ under transgenerational, the resolver SHALL return (None, None, parent_id).

    F1+ children inherit the substrate via a separate kwarg path
    into ``fitness.evaluate`` (a follow-up addition), not via
    per-child weight-IO. Matches Baldwin's trait-only flow shape.
    """
    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = TransgenerationalInheritance()
    loop._selected_parent_ids = ["f0-elite"]
    warm_start, capture, inherited_from = loop._resolve_per_child_inheritance(
        child_idx=0,
        gen=1,
        gid="gid-b",
    )
    assert warm_start is None
    assert capture is None
    assert inherited_from == "f0-elite"


def test_resolve_per_child_inheritance_transgenerational_f1_without_elite(tmp_path: Path) -> None:
    """Under transgenerational at gen 1+ with no elite yet, inherited_from SHALL be empty."""
    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = TransgenerationalInheritance()
    loop._selected_parent_ids = []  # No elite recorded.
    warm_start, capture, inherited_from = loop._resolve_per_child_inheritance(
        child_idx=0,
        gen=1,
        gid="gid-b",
    )
    assert warm_start is None
    assert capture is None
    assert inherited_from == ""


# ---------------------------------------------------------------------------
# F0 substrate extraction pipeline: graceful behaviour with missing weights
# ---------------------------------------------------------------------------


def test_f0_extraction_missing_weights_file_logs_warning_and_skips(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the F0 elite ``.pt`` is missing, extraction SHALL log a warning and skip the save.

    Defensive against unexpected on-disk state (e.g., a worker crashed
    before writing weights). The loop continues to F1 with no
    substrate; downstream worker integration (a follow-up commit)
    will detect the missing ``.tei.pt`` and handle it.
    """
    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = TransgenerationalInheritance()
    # The elite ``.pt`` is intentionally NOT written, simulating the
    # crash-before-capture failure mode.
    elite_id = "elite-missing"
    # Build synthetic gen_ids + solutions matching what the loop
    # would provide; the elite's index in gen_ids points at the
    # element of solutions to decode.
    gen_ids = [elite_id]
    solutions = [np.zeros(loop.encoder.genome_dim(loop.sim_config), dtype=np.float32)]
    with caplog.at_level(logging.WARNING):
        loop._run_f0_substrate_extraction(
            elite_id=elite_id,
            gen_ids=gen_ids,
            solutions=solutions,
        )
    assert "elite weights at" in caplog.text or "skipping substrate save" in caplog.text
    # No .tei.pt should have been written.
    substrate_path = tmp_path / "inheritance" / "gen-000" / f"genome-{elite_id}.tei.pt"
    assert not substrate_path.exists()


def test_f0_extraction_elite_not_in_gen_ids_logs_warning_and_skips(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the elite_id isn't in gen_ids, extraction SHALL log and skip.

    Defensive against a hypothetical state where ``select_parents``
    returns an ID that wasn't in the population (shouldn't happen,
    but the helper guards against it cleanly).
    """
    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = TransgenerationalInheritance()
    gen_ids = ["a", "b", "c"]
    solutions = [np.zeros(loop.encoder.genome_dim(loop.sim_config), dtype=np.float32)] * 3
    with caplog.at_level(logging.WARNING):
        loop._run_f0_substrate_extraction(
            elite_id="not-in-gen-ids",
            gen_ids=gen_ids,
            solutions=solutions,
        )
    assert "not found in gen_ids" in caplog.text


# ---------------------------------------------------------------------------
# F0 substrate extraction pipeline: happy path
# ---------------------------------------------------------------------------


def test_f0_extraction_happy_path_writes_substrate_and_gcs_pt_files(tmp_path: Path) -> None:
    """The F0 pipeline SHALL write the ``.tei.pt`` substrate AND delete every ``.pt`` weight file.

    Synthetic happy-path test: write fake F0 ``.pt`` weight files for
    the elite + two non-elite children (simulating what F0 workers
    would produce via the ``weight_capture_path`` kwarg), then invoke
    the pipeline. Assert (a) the elite's ``.tei.pt`` is written, (b)
    all three ``.pt`` files are deleted, (c) ``self._tei_f0_substrate_path``
    is populated to the substrate path.
    """
    from quantumnematode.brain.weights import save_weights

    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = TransgenerationalInheritance()
    elite_id = "elite-happy"
    other_id_a = "other-a"
    other_id_b = "other-b"
    gen_ids = [elite_id, other_id_a, other_id_b]

    # Build solutions: just zero-arrays since the elite weights load
    # below overwrites whatever the encoder.decode initialised.
    dim = loop.encoder.genome_dim(loop.sim_config)
    solutions = [np.zeros(dim, dtype=np.float32) for _ in gen_ids]

    # Pre-write fake .pt weight files at the canonical capture paths
    # for all three F0 children, simulating what F0 workers would
    # produce via the weight_capture_path kwarg.
    gen_dir = tmp_path / "inheritance" / "gen-000"
    gen_dir.mkdir(parents=True, exist_ok=True)
    for gid in gen_ids:
        # Construct a real brain via the encoder + save its weights.
        # The exact weights don't matter for this test (we're testing
        # the pipeline mechanics, not the extraction-algorithm output);
        # what matters is that load_weights succeeds on the file.
        from quantumnematode.evolution.genome import Genome

        real_genome = Genome(
            params=np.zeros(dim, dtype=np.float32),
            genome_id=gid,
            parent_ids=[],
            generation=0,
            birth_metadata={},
        )
        scratch_brain = loop.encoder.decode(real_genome, loop.sim_config, seed=0)
        save_weights(scratch_brain, gen_dir / f"genome-{gid}.pt")

    # Verify pre-state: 3 .pt files, no .tei.pt.
    pt_files_before = sorted(gen_dir.glob("*.pt"))
    assert len(pt_files_before) == 3
    assert not (gen_dir / f"genome-{elite_id}.tei.pt").exists()

    # Run the pipeline.
    loop._run_f0_substrate_extraction(
        elite_id=elite_id,
        gen_ids=gen_ids,
        solutions=solutions,
    )

    # Post-state: only the elite's .tei.pt remains.
    substrate_path = gen_dir / f"genome-{elite_id}.tei.pt"
    assert substrate_path.exists()
    # All three .pt files are deleted.
    pt_files_after = sorted(gen_dir.glob("genome-*.pt"))
    # Filter out .tei.pt (which has .pt suffix in path.suffix but ends in .tei.pt).
    pt_files_after_excluding_tei = [p for p in pt_files_after if not p.name.endswith(".tei.pt")]
    assert pt_files_after_excluding_tei == []
    # The instance attribute is populated for downstream F1+ worker dispatch.
    assert loop._tei_f0_substrate_path == substrate_path


def test_f0_extraction_persists_substrate_path_in_checkpoint(tmp_path: Path) -> None:
    """``self._tei_f0_substrate_path`` SHALL round-trip through checkpoint save/load.

    Persistence contract from the evolution-framework spec: the
    F0 substrate path must survive a checkpoint cycle so resume
    from gen 1+ can recover the path without re-running F0.

    Uses ``inheritance="none"`` end-to-end on both save and load
    so the inheritance-mismatch validator doesn't fire (the
    persistence behaviour being tested is independent of which
    strategy is active; setting the attribute directly bypasses
    the actual F0 pipeline). The contract we're testing is:
    if the attribute is set, save/load preserves it.
    """
    loop = _make_baseline_loop(tmp_path)
    # Directly set the attribute as if the F0 pipeline had run.
    original_path = tmp_path / "inheritance" / "gen-000" / "genome-elite-persist.tei.pt"
    loop._tei_f0_substrate_path = original_path

    # Save the checkpoint.
    loop._save_checkpoint()

    # Load into a fresh loop with the same config (so the validator passes).
    fresh_loop = _make_baseline_loop(tmp_path)
    fresh_loop._load_checkpoint(loop._checkpoint_path)

    assert fresh_loop._tei_f0_substrate_path == original_path


def test_checkpoint_load_defaults_tei_f0_substrate_path_to_none(tmp_path: Path) -> None:
    """Pre-TEI checkpoints (no ``tei_f0_substrate_path`` key) SHALL load with the attr = None.

    Backwards-compatibility contract: the field is additive, so
    old v3 checkpoints (which predate the TEI cascade) can still
    be loaded. The loader uses ``payload.get(..., None)`` so a
    missing key produces ``None`` rather than KeyError.
    """
    import pickle as _pickle

    loop = _make_baseline_loop(tmp_path)
    # Save a checkpoint without the new key (simulating a v3 pickle).
    loop._tei_f0_substrate_path = None
    loop._save_checkpoint()
    # Remove the new key from the on-disk pickle to simulate a pre-TEI
    # checkpoint that doesn't contain the field at all.
    with loop._checkpoint_path.open("rb") as handle:
        payload = _pickle.load(handle)  # noqa: S301 — trusted local test fixture
    payload.pop("tei_f0_substrate_path", None)
    with loop._checkpoint_path.open("wb") as handle:
        _pickle.dump(payload, handle)

    fresh_loop = _make_baseline_loop(tmp_path)
    fresh_loop._load_checkpoint(loop._checkpoint_path)
    assert fresh_loop._tei_f0_substrate_path is None


# ---------------------------------------------------------------------------
# Sanity check: existing non-TEI helpers are unaffected
# ---------------------------------------------------------------------------


def test_existing_helpers_unchanged_for_lamarckian(tmp_path: Path) -> None:
    """Lamarckian path SHALL NOT be affected by the new substrate helper.

    Sanity check: ``_substrate_inheritance_active()`` returns False
    for Lamarckian, and ``_resolve_per_child_inheritance`` still
    follows the weights branch (returns ``(None, capture_path,
    parent_id)`` for gen 1+).
    """
    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = LamarckianInheritance()
    assert loop._substrate_inheritance_active() is False
    # Lamarckian at gen 1+ with a selected parent returns the warm-start path,
    # so we use gen=0 (no warm-start yet) to test the capture-only shape.
    loop._selected_parent_ids = []
    warm_start, capture, inherited_from = loop._resolve_per_child_inheritance(
        child_idx=0,
        gen=0,
        gid="gid-a",
    )
    assert warm_start is None  # gen 0: no parent yet
    assert capture is not None
    assert capture.name == "genome-gid-a.pt"
    assert inherited_from == ""
