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


# ---------------------------------------------------------------------------
# B1 / B2 regression: F0 extraction forwards YAML bias_network spec
# + extraction_seed (the M6.9+ correctness gate).
# ---------------------------------------------------------------------------


def _attach_transgenerational_config_with_bias_network(
    loop: EvolutionLoop,
    *,
    extraction_seed: int = 424242,
    hidden_dim: int = 8,
    input_features: tuple[str, ...] = (
        "predator_gradient_strength",
        "predator_gradient_direction_sin",
        "food_gradient_strength",
    ),
) -> None:
    """Attach a minimal ``transgenerational`` config block with ``bias_network`` set.

    Bypasses the inheritance-pairing validator (``enabled=True`` would
    normally require ``inheritance="transgenerational"``) by setting
    ``enabled=False`` and overriding the loop's inheritance instance
    afterwards. This is the same monkey-patch pattern used by
    ``_make_baseline_loop``: the validator runs at config construction,
    not at every field read.
    """
    from quantumnematode.utils.config_loader import (
        BiasNetworkConfig,
        LawnScheduleEntry,
        TransgenerationalConfig,
    )

    bias_network = BiasNetworkConfig(
        hidden_dim=hidden_dim,
        activation="tanh",
        input_features=list(input_features),
    )
    tg_cfg = TransgenerationalConfig(
        enabled=False,
        decay_factor=0.6,
        decay_shape="geometric",
        extraction_seed=extraction_seed,
        lawn_schedule=[
            LawnScheduleEntry(
                generation=0,
                pathogen_lawns_enabled=True,
                ppo_train_episodes=10,
            ),
            LawnScheduleEntry(
                generation=1,
                pathogen_lawns_enabled=True,
                ppo_train_episodes=10,
            ),
        ],
        bias_network=bias_network,
    )
    loop.evolution_config = loop.evolution_config.model_copy(
        update={"transgenerational": tg_cfg},
    )


def test_f0_extraction_forwards_bias_network_spec_from_config(
    tmp_path: Path,
) -> None:
    """B1 regression: ``cfg.transgenerational.bias_network`` MUST flow to ``extract_from_brain``.

    When the YAML configures a ``bias_network`` sub-block, the F0
    extractor receives ``bias_network_spec`` + ``input_features``
    matching it.

    Without this wiring, the M6.9+ ``tei_on`` arm silently regresses
    to M6 behaviour (constant ``logit_bias``, ``bias_network=None``)
    and the original "3-of-4 bit-identical substrate" attractor
    re-appears across calibration seeds. This test pins the call-site
    contract using a spy on ``extract_from_brain``.
    """
    from unittest.mock import patch

    from quantumnematode.brain.weights import save_weights
    from quantumnematode.evolution.genome import Genome

    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = TransgenerationalInheritance()
    _attach_transgenerational_config_with_bias_network(
        loop,
        extraction_seed=98765,
        hidden_dim=4,
        input_features=("predator_gradient_strength", "food_gradient_strength"),
    )

    elite_id = "elite-b1"
    gen_ids = [elite_id]
    dim = loop.encoder.genome_dim(loop.sim_config)
    solutions = [np.zeros(dim, dtype=np.float32) for _ in gen_ids]

    gen_dir = tmp_path / "inheritance" / "gen-000"
    gen_dir.mkdir(parents=True, exist_ok=True)
    real_genome = Genome(
        params=np.zeros(dim, dtype=np.float32),
        genome_id=elite_id,
        parent_ids=[],
        generation=0,
        birth_metadata={},
    )
    scratch_brain = loop.encoder.decode(real_genome, loop.sim_config, seed=0)
    save_weights(scratch_brain, gen_dir / f"genome-{elite_id}.pt")

    captured_kwargs: dict = {}
    # Bind the REAL implementation BEFORE patching, otherwise the
    # delegating call inside the spy resolves back to the mock and
    # recurses infinitely.
    from quantumnematode.agent.transgenerational_memory import (
        extract_from_brain as _real_extract,
    )

    def _spy_extract(*args: object, **kwargs: object) -> object:
        captured_kwargs.update(kwargs)
        return _real_extract(*args, **kwargs)  # type: ignore[arg-type]

    with patch(
        "quantumnematode.agent.transgenerational_memory.extract_from_brain",
        side_effect=_spy_extract,
    ):
        loop._run_f0_substrate_extraction(
            elite_id=elite_id,
            gen_ids=gen_ids,
            solutions=solutions,
        )

    # B1: bias_network_spec is forwarded with the YAML-configured shape.
    assert "bias_network_spec" in captured_kwargs
    spec = captured_kwargs["bias_network_spec"]
    assert spec is not None
    assert spec["input_dim"] == 2  # len(input_features) above
    assert spec["hidden_dim"] == 4
    assert spec["activation"] == "tanh"
    assert spec["output_dim"] == scratch_brain.num_actions  # type: ignore[attr-defined]

    # B1: input_features is forwarded as a tuple matching the YAML.
    assert captured_kwargs["input_features"] == (
        "predator_gradient_strength",
        "food_gradient_strength",
    )

    # B2: extraction_seed reads from the YAML, NOT the hard-coded 424242.
    assert captured_kwargs["rng_seed"] == 98765


def test_f0_extraction_legacy_path_when_bias_network_absent(tmp_path: Path) -> None:
    """B1 negative: when ``cfg.transgenerational.bias_network is None``, the legacy M6 path runs.

    Forward-compatibility / backwards-compat: a transgenerational
    config without an explicit ``bias_network`` block falls back to
    the M6 constant ``logit_bias`` extraction path. The call-site
    MUST pass ``bias_network_spec=None`` so ``extract_from_brain``
    dispatches the legacy branch (``input_features=()``).
    """
    from unittest.mock import patch

    from quantumnematode.brain.weights import save_weights
    from quantumnematode.evolution.genome import Genome
    from quantumnematode.utils.config_loader import (
        LawnScheduleEntry,
        TransgenerationalConfig,
    )

    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = TransgenerationalInheritance()
    # Configure transgenerational WITHOUT a bias_network sub-block.
    tg_cfg = TransgenerationalConfig(
        enabled=False,
        decay_factor=0.6,
        extraction_seed=11111,
        lawn_schedule=[
            LawnScheduleEntry(
                generation=0,
                pathogen_lawns_enabled=True,
                ppo_train_episodes=10,
            ),
            LawnScheduleEntry(
                generation=1,
                pathogen_lawns_enabled=True,
                ppo_train_episodes=10,
            ),
        ],
        # bias_network defaults to None.
    )
    loop.evolution_config = loop.evolution_config.model_copy(
        update={"transgenerational": tg_cfg},
    )

    elite_id = "elite-legacy"
    gen_ids = [elite_id]
    dim = loop.encoder.genome_dim(loop.sim_config)
    solutions = [np.zeros(dim, dtype=np.float32) for _ in gen_ids]
    gen_dir = tmp_path / "inheritance" / "gen-000"
    gen_dir.mkdir(parents=True, exist_ok=True)
    real_genome = Genome(
        params=np.zeros(dim, dtype=np.float32),
        genome_id=elite_id,
        parent_ids=[],
        generation=0,
        birth_metadata={},
    )
    scratch_brain = loop.encoder.decode(real_genome, loop.sim_config, seed=0)
    save_weights(scratch_brain, gen_dir / f"genome-{elite_id}.pt")

    captured_kwargs: dict = {}
    from quantumnematode.agent.transgenerational_memory import (
        extract_from_brain as _real_extract,
    )

    def _spy_extract(*args: object, **kwargs: object) -> object:
        captured_kwargs.update(kwargs)
        return _real_extract(*args, **kwargs)  # type: ignore[arg-type]

    with patch(
        "quantumnematode.agent.transgenerational_memory.extract_from_brain",
        side_effect=_spy_extract,
    ):
        loop._run_f0_substrate_extraction(
            elite_id=elite_id,
            gen_ids=gen_ids,
            solutions=solutions,
        )

    # Legacy path: bias_network_spec is None, input_features is empty.
    assert captured_kwargs["bias_network_spec"] is None
    assert captured_kwargs["input_features"] == ()
    # extraction_seed still threads from YAML.
    assert captured_kwargs["rng_seed"] == 11111


def test_f0_substrate_diversity_across_seeds_with_bias_network(tmp_path: Path) -> None:
    """B1 + B2 end-to-end: four ``extraction_seed`` values produce diverse substrates.

    This is the regression gate against the original M6 incident
    (3-of-4 calibration seeds extracting bit-identical substrates).
    Once B1 forwards the YAML ``bias_network`` spec AND B2 forwards
    the YAML ``extraction_seed``, four runs with distinct seeds
    SHALL produce substrates whose pairwise CoV (per the T2
    tripwire definition) is non-zero (a degenerate substrate
    collapse would re-appear here first).
    """
    import torch
    from quantumnematode.agent.transgenerational_memory import (
        TransgenerationalMemory,
    )
    from quantumnematode.agent.transgenerational_memory import (
        load as load_substrate,
    )
    from quantumnematode.brain.weights import save_weights
    from quantumnematode.evolution.genome import Genome

    substrates: list[TransgenerationalMemory] = []
    for seed_idx, extraction_seed in enumerate((42, 43, 44, 45)):
        seed_tmp = tmp_path / f"seed-{extraction_seed}"
        seed_tmp.mkdir()
        loop = _make_baseline_loop(seed_tmp)
        loop.inheritance = TransgenerationalInheritance()
        _attach_transgenerational_config_with_bias_network(
            loop,
            extraction_seed=extraction_seed,
            hidden_dim=8,
        )

        elite_id = f"elite-{seed_idx}"
        gen_ids = [elite_id]
        dim = loop.encoder.genome_dim(loop.sim_config)
        solutions = [np.zeros(dim, dtype=np.float32) for _ in gen_ids]
        gen_dir = seed_tmp / "inheritance" / "gen-000"
        gen_dir.mkdir(parents=True, exist_ok=True)
        real_genome = Genome(
            params=np.zeros(dim, dtype=np.float32),
            genome_id=elite_id,
            parent_ids=[],
            generation=0,
            birth_metadata={},
        )
        scratch_brain = loop.encoder.decode(real_genome, loop.sim_config, seed=0)
        save_weights(scratch_brain, gen_dir / f"genome-{elite_id}.pt")

        loop._run_f0_substrate_extraction(
            elite_id=elite_id,
            gen_ids=gen_ids,
            solutions=solutions,
        )
        assert loop._tei_f0_substrate_path is not None
        substrates.append(load_substrate(loop._tei_f0_substrate_path))

    # All four substrates SHALL have a populated bias_network (M6.9+ path).
    assert all(s.bias_network is not None for s in substrates)

    # Pairwise CoV is non-zero on the flattened state_dict — substrate
    # is NOT bit-identical across seeds. The original M6 attractor
    # signature would surface as CoV == 0 for at least one pair here.
    def _flat(s: TransgenerationalMemory) -> torch.Tensor:
        assert s.bias_network is not None
        return torch.cat(
            [t.flatten().to(torch.float64) for t in s.bias_network.state_dict().values()],
        )

    flat = [_flat(s) for s in substrates]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            diff = torch.linalg.vector_norm(flat[i] - flat[j])
            assert diff.item() > 0.0, (
                f"substrates for seed pairs ({i}, {j}) are bit-identical — "
                "B1/B2 fix is incomplete or extraction is degenerate."
            )


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


def test_build_f0_probe_params_matches_brain_input_dim_under_stam(tmp_path: Path) -> None:
    """Probe ``BrainParams`` MUST produce features matching ``brain.input_dim``.

    Regression for an F0 extraction crash on configs with STAM + multiple
    non-STAM modules. The brain's ``input_dim`` is built from the
    inferred STAM dim (e.g., 7 for a 2-channel env), but the
    ``STAMSensoryModule`` registry instance has ``classical_dim=11``
    (4-channel default). When the probe ``BrainParams`` is constructed
    without an explicit ``stam_state``, the runtime feature pipeline
    emits 11 zeros for STAM — total features don't match the brain's
    ``feature_norm.normalized_shape`` and ``torch.layer_norm`` raises.
    The fix is in ``_build_f0_probe_params(brain=brain)`` which
    derives the STAM dim from the brain's known ``input_dim``.

    Uses a minimal in-test sim_config (not the live transgenerational
    pilot YAML) so future YAML edits don't silently change test
    coverage. The bug requires STAM + multiple non-STAM channels to
    surface; ``[food_chemotaxis, nociception]`` is the minimum
    reproducer (1 food channel + 1 predator channel → 2-channel STAM
    of dim 7, vs the registry default of 11).
    """
    # Start from a stable LSTMPPO+klinotaxis scenario YAML, then patch
    # the brain block to add ``nociception`` (the bug-trigger). The
    # scenario YAML doesn't have a predator block; the brain just
    # produces zero nociception features at runtime — that's fine for
    # this dim-matching test (we're not exercising the predator logic).
    from quantumnematode.brain.modules import ModuleName, extract_classical_features
    from quantumnematode.evolution.encoders import (
        HyperparameterEncoder,
        build_birth_metadata,
    )
    from quantumnematode.evolution.genome import Genome
    from quantumnematode.utils.config_loader import ParamSchemaEntry

    base_yaml = PROJECT_ROOT / "configs" / "scenarios" / "foraging" / "lstmppo_small_klinotaxis.yml"
    assert base_yaml.exists(), f"Base scenario YAML not found: {base_yaml}"
    sim_config = load_simulation_config(str(base_yaml))
    # Inject nociception so STAM has 2 channels (food + predator) — the
    # minimum reproducer for the inferred-STAM-vs-registry-STAM bug.
    assert sim_config.brain is not None
    # ``sensory_modules`` exists on LSTMPPO/MLPPPO brain configs but not
    # on every brain in the BrainConfig union; getattr() with the known
    # default keeps pyright quiet.
    existing_modules = list(getattr(sim_config.brain.config, "sensory_modules", []) or [])
    patched_modules = [*existing_modules, ModuleName.NOCICEPTION]
    new_brain_cfg = sim_config.brain.config.model_copy(
        update={"sensory_modules": patched_modules},
    )
    new_brain_container = sim_config.brain.model_copy(update={"config": new_brain_cfg})
    # Add a minimal hyperparam_schema so HyperparameterEncoder works.
    sim_config = sim_config.model_copy(
        update={
            "brain": new_brain_container,
            "hyperparam_schema": [
                ParamSchemaEntry(name="actor_lr", type="float", bounds=(1e-5, 1e-3)),
            ],
        },
    )

    encoder = HyperparameterEncoder()
    # Decode a fresh brain with an arbitrary genome value — we're only
    # exercising brain construction, not training.
    genome = Genome(
        params=np.array([1e-4], dtype=np.float32),
        genome_id="probe-test",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )
    brain = encoder.decode(genome, sim_config, seed=42)

    # Construct a minimal EvolutionLoop just so we can call the helper.
    encoder_for_loop = HyperparameterEncoder()
    fitness = EpisodicSuccessRate()
    ecfg = EvolutionConfig(
        algorithm="cmaes",
        population_size=2,
        generations=1,
        episodes_per_eval=1,
        learn_episodes_per_eval=1,
    )
    optimizer = CMAESOptimizer(
        num_params=encoder_for_loop.genome_dim(sim_config),
        population_size=2,
        sigma0=ecfg.sigma0,
        seed=42,
    )
    loop = EvolutionLoop(
        optimizer=optimizer,
        encoder=encoder_for_loop,
        fitness=fitness,
        sim_config=sim_config,
        evolution_config=ecfg,
        output_dir=tmp_path,
        rng=np.random.default_rng(42),
        log_level=logging.WARNING,
    )

    probe_params_list = loop._build_f0_probe_params(brain=brain)
    assert len(probe_params_list) > 0, "Probe params list must be non-empty"

    # Each probe MUST produce features matching brain.input_dim, otherwise
    # ``feature_norm`` raises a RuntimeError in ``run_brain``.
    # ``sensory_modules`` and ``input_dim`` are LSTMPPO-specific (not on
    # the Brain protocol); cast via getattr to keep pyright quiet.
    sensory_modules = brain.sensory_modules  # type: ignore[attr-defined]
    input_dim = int(brain.input_dim)  # type: ignore[attr-defined]
    # Sanity-check that this config DOES exercise the bug-relevant
    # state — STAM module is in the brain's effective modules and the
    # inferred STAM dim is < the registry default of 11 (the gap that
    # the probe fix bridges).
    assert ModuleName.STAM in sensory_modules, (
        "Test fixture must include STAM (auto-enabled by klinotaxis); "
        "otherwise the regression isn't being exercised."
    )
    for idx, probe in enumerate(probe_params_list):
        features = extract_classical_features(probe, sensory_modules)
        assert features.shape[0] == input_dim, (
            f"Probe #{idx} produced features of shape {features.shape}, "
            f"but brain.input_dim is {input_dim}. "
            f"This will trip ``feature_norm`` during ``run_brain``."
        )


def test_build_f0_probe_params_without_brain_preserves_legacy_shape(tmp_path: Path) -> None:
    """Without the optional ``brain`` arg, the helper SHALL still return probe params.

    Backwards-compatibility check: existing callers / tests that don't
    pass ``brain`` keep producing the legacy three-probe sequence
    (with ``stam_state=None``). The legacy path is fine for brains
    without STAM and for foraging-only configs.
    """
    loop = _make_baseline_loop(tmp_path)
    loop.inheritance = TransgenerationalInheritance()
    probe_params_list = loop._build_f0_probe_params()
    assert len(probe_params_list) == 3
    # No brain → no stam_state derivation → legacy None default.
    for probe in probe_params_list:
        assert probe.stam_state is None


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
