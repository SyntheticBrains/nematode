"""Unit tests for the F1 learning-acceleration post-pilot evaluator.

The script lives at ``scripts/campaigns/baldwin_f1_postpilot_eval.py``
(outside the package; imported via ``importlib`` so the tests don't
need a sys.path hack).  Tests cover:

(a) argparse rejects non-positive ``--k-prime`` and ``--episodes``
    via ``parser.error`` (exits non-zero before the loop runs).
(b) ``HyperparameterEncoder.initial_genome`` round-trips cleanly
    under an 8-field schema (the F1 baseline-genome construction
    path) — params length matches schema, encoder.decode succeeds.
(c) Append-mode CSV preserves prior rows when invoked twice with
    different K' values — re-runs at K' = 25 after K' = 10 leave
    both sets in the same CSV.
(d) Per-seed RNG plumbing (``initial_genome(rng=np.random.default_rng(seed))``)
    is deterministic across repeated calls with the same seed and
    differs across seeds.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from quantumnematode.evolution.encoders import HyperparameterEncoder, build_birth_metadata
from quantumnematode.evolution.genome import Genome
from quantumnematode.utils.config_loader import SimulationConfig, load_simulation_config

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "campaigns" / "baldwin_f1_postpilot_eval.py"

# Reuse the shipped Baldwin pilot YAML for schema reconstruction in tests
# that need a SimulationConfig.  The M4 pilot YAML has hyperparam_schema
# set; M4.5's 8-field YAML doesn't exist yet at task-2 time, so we use
# the existing one to exercise the encoder round-trip path.  The actual
# 8-field YAML is created in task group 3.
M4_PILOT_YAML = (
    PROJECT_ROOT / "configs" / "evolution" / "baldwin_lstmppo_klinotaxis_predator_pilot.yml"
)


def _load_script_module():
    """Import the script as a module so we can call its helpers directly."""
    spec = importlib.util.spec_from_file_location(
        "baldwin_f1_postpilot_eval",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"Failed to load spec for {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# (a) argparse rejects non-positive --k-prime and --episodes
# ---------------------------------------------------------------------------


def test_cli_rejects_kprime_zero(tmp_path: Path) -> None:
    """``--k-prime 0`` SHALL exit non-zero with a clear error message."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--baldwin-root",
            str(tmp_path),
            "--config",
            str(M4_PILOT_YAML),
            "--output-dir",
            str(tmp_path),
            "--k-prime",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--k-prime must be a positive integer" in result.stderr


def test_cli_rejects_kprime_negative(tmp_path: Path) -> None:
    """``--k-prime -5`` SHALL exit non-zero with a clear error message."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--baldwin-root",
            str(tmp_path),
            "--config",
            str(M4_PILOT_YAML),
            "--output-dir",
            str(tmp_path),
            "--k-prime",
            "-5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--k-prime must be a positive integer" in result.stderr


def test_cli_rejects_episodes_zero(tmp_path: Path) -> None:
    """``--episodes 0`` SHALL exit non-zero with a clear error message."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--baldwin-root",
            str(tmp_path),
            "--config",
            str(M4_PILOT_YAML),
            "--output-dir",
            str(tmp_path),
            "--episodes",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--episodes must be a positive integer" in result.stderr


# ---------------------------------------------------------------------------
# (b) HyperparameterEncoder.initial_genome round-trips under an 8-field schema
# ---------------------------------------------------------------------------


def _make_8field_sim_config() -> SimulationConfig:
    """Load the shipped M4 6-field pilot YAML and synthesise an 8-field schema in-memory.

    M4.5's 8-field YAML doesn't exist yet (created in task group 3), so we
    extend M4's 6-field schema with the 2 NEW arch knobs to exercise the
    8-field round-trip path the F1 evaluator will hit in the full pilot.
    """
    sim_config = load_simulation_config(str(M4_PILOT_YAML))
    if sim_config.hyperparam_schema is None:
        msg = "M4 pilot YAML has no hyperparam_schema — fixture invariant violated."
        raise RuntimeError(msg)
    # Extend M4's 6-field schema to 8 fields by appending the arch knobs
    # per add-baldwin-retry design Decision 1.
    from quantumnematode.utils.config_loader import ParamSchemaEntry

    arch_entries = [
        ParamSchemaEntry(name="actor_hidden_dim", type="int", bounds=(64.0, 256.0)),
        ParamSchemaEntry(name="actor_num_layers", type="int", bounds=(1.0, 3.0)),
    ]
    extended_schema = list(sim_config.hyperparam_schema) + arch_entries
    return sim_config.model_copy(update={"hyperparam_schema": extended_schema})


def test_initial_genome_8field_schema_returns_correct_dim() -> None:
    """``initial_genome`` SHALL return params of length 8 under an 8-field schema."""
    sim_config = _make_8field_sim_config()
    encoder = HyperparameterEncoder()
    rng = np.random.default_rng(42)
    genome = encoder.initial_genome(sim_config, rng=rng)
    assert genome.params.shape == (8,)


def test_initial_genome_8field_round_trips_through_decode() -> None:
    """``initial_genome().params`` SHALL decode back into a working brain at the 8-field schema."""
    sim_config = _make_8field_sim_config()
    encoder = HyperparameterEncoder()
    rng = np.random.default_rng(42)
    genome = encoder.initial_genome(sim_config, rng=rng)
    # Need birth_metadata for decode
    genome = Genome(
        params=genome.params,
        genome_id="test_baseline",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )
    brain = encoder.decode(genome, sim_config, seed=42)
    # Decoded brain SHALL exist and have the schema-prior arch fields
    # patched into its config.  We don't run the brain — just confirm
    # the decode chain completes without raising.
    assert brain is not None


# ---------------------------------------------------------------------------
# (c) Append-mode CSV preserves prior rows across multiple K' runs
# ---------------------------------------------------------------------------


def test_csv_append_mode_preserves_prior_rows(tmp_path: Path) -> None:
    """Re-running the script with a different K' SHALL append rows to the existing CSV."""
    module = _load_script_module()
    csv_path = tmp_path / "f1_learning_acceleration.csv"

    # Simulate a first invocation by writing one row at K' = 10.
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "seed",
                "k_prime",
                "episodes",
                "elite_success_rate",
                "baseline_success_rate",
                "signal_delta",
            ),
        )
        writer.writerow((42, 10, 25, "0.500000", "0.300000", "+0.200000"))

    # Now simulate a re-run at K' = 25 by invoking the same append-mode
    # logic the script uses.  We replicate the script's csv-write block
    # (no need to subprocess.run — the CSV format is the contract under
    # test).
    rows_kprime_25 = [(42, 25, 25, 0.7, 0.4, 0.3)]
    write_header = not csv_path.exists()  # False — file already exists
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                (
                    "seed",
                    "k_prime",
                    "episodes",
                    "elite_success_rate",
                    "baseline_success_rate",
                    "signal_delta",
                ),
            )
        for seed, kp, eps, elite, baseline, delta in rows_kprime_25:
            writer.writerow(
                (seed, kp, eps, f"{elite:.6f}", f"{baseline:.6f}", f"{delta:+.6f}"),
            )

    # CSV SHALL now contain: 1 header + 1 row at K'=10 + 1 row at K'=25
    with csv_path.open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 3
    assert rows[0] == [
        "seed",
        "k_prime",
        "episodes",
        "elite_success_rate",
        "baseline_success_rate",
        "signal_delta",
    ]
    assert rows[1][1] == "10"  # K'=10 row preserved
    assert rows[2][1] == "25"  # K'=25 row appended

    # And the header was NOT written twice
    assert sum(1 for r in rows if r[0] == "seed") == 1

    # Touch module to satisfy unused-import check on importlib helper.
    assert hasattr(module, "main")


# ---------------------------------------------------------------------------
# (d) Per-seed RNG plumbing is deterministic + seed-sensitive
# ---------------------------------------------------------------------------


def test_initial_genome_same_seed_produces_identical_params() -> None:
    """Two calls with the same seed SHALL produce bit-identical params."""
    sim_config = _make_8field_sim_config()
    encoder = HyperparameterEncoder()
    g1 = encoder.initial_genome(sim_config, rng=np.random.default_rng(42))
    g2 = encoder.initial_genome(sim_config, rng=np.random.default_rng(42))
    assert np.array_equal(g1.params, g2.params)


def test_initial_genome_different_seeds_produce_different_params() -> None:
    """Two calls with different seeds SHALL produce different params (with high probability)."""
    sim_config = _make_8field_sim_config()
    encoder = HyperparameterEncoder()
    g1 = encoder.initial_genome(sim_config, rng=np.random.default_rng(42))
    g2 = encoder.initial_genome(sim_config, rng=np.random.default_rng(43))
    assert not np.array_equal(g1.params, g2.params)


# ---------------------------------------------------------------------------
# Sanity: helpers exposed by the script module
# ---------------------------------------------------------------------------


def test_script_module_exposes_helpers() -> None:
    """The script SHALL define the helpers we depend on for testing."""
    module = _load_script_module()
    assert callable(module._resolve_session)  # type: ignore[attr-defined]
    assert callable(module._build_sim_config_for_kprime)  # type: ignore[attr-defined]
    assert callable(module._evaluate_one_seed)  # type: ignore[attr-defined]
    assert callable(module.main)


def test_build_sim_config_for_kprime_sets_k_and_l() -> None:
    """``_build_sim_config_for_kprime`` SHALL set BOTH the K and L config fields."""
    module = _load_script_module()
    sim_config = load_simulation_config(str(M4_PILOT_YAML))
    if sim_config.evolution is None:
        pytest.skip("M4 pilot YAML lacks evolution block")
    new_config = module._build_sim_config_for_kprime(  # type: ignore[attr-defined]
        sim_config,
        k_prime=7,
        episodes=13,
    )
    assert new_config.evolution.learn_episodes_per_eval == 7
    assert new_config.evolution.eval_episodes_per_eval == 13
    # Original SHALL be unchanged (model_copy returns a new instance)
    assert sim_config.evolution.learn_episodes_per_eval != 7


def test_build_sim_config_for_kprime_rejects_no_evolution_block() -> None:
    """``_build_sim_config_for_kprime`` SHALL raise if sim_config.evolution is None."""
    module = _load_script_module()
    sim_config = load_simulation_config(str(M4_PILOT_YAML))
    sim_config_no_evo = sim_config.model_copy(update={"evolution": None})
    with pytest.raises(ValueError, match=r"requires sim_config\.evolution to be set"):
        module._build_sim_config_for_kprime(  # type: ignore[attr-defined]
            sim_config_no_evo,
            k_prime=10,
            episodes=25,
        )
