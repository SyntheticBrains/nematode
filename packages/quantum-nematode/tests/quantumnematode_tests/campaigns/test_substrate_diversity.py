"""Tests for the M6.9+ substrate-diversity tripwire script (T2 + T4).

Covers the pairwise CoV computation, the threshold pass/fail semantics,
the degenerate (bit-identical) substrate detection, and the empty/
single-seed degenerate input rejection.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from quantumnematode.agent.transgenerational_memory import (
    TransgenerationalMemory,
)
from quantumnematode.agent.transgenerational_memory import (
    save as save_substrate,
)
from torch import nn

if TYPE_CHECKING:
    from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_PATH = PROJECT_ROOT / "scripts/campaigns/m69_substrate_diversity.py"


def _load_script_module() -> ModuleType:
    """Dynamically load the diversity script as a module."""
    spec = importlib.util.spec_from_file_location("m69_substrate_diversity", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_bias_network(*, seed: int, hidden_dim: int = 8) -> nn.Sequential:
    """Build a deterministic 3 -> hidden -> 4 MLP with seed-reproducible weights."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(3, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 4),
    )
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=gen))
    for p in net.parameters():
        p.requires_grad = False
    return net


def _make_substrate(*, seed: int, hidden_dim: int = 8) -> TransgenerationalMemory:
    """Build a substrate with a deterministic per-seed bias_network."""
    return TransgenerationalMemory(
        logit_bias=torch.zeros(4),
        lineage_depth=0,
        source_genome_id="test",
        bias_network=_make_bias_network(seed=seed, hidden_dim=hidden_dim),
        input_features=(
            "predator_gradient_strength",
            "predator_gradient_direction_sin",
            "food_gradient_strength",
        ),
    )


# ---------------------------------------------------------------------------
# pairwise_cov + compute_pairwise_cov_matrix
# ---------------------------------------------------------------------------


def test_pairwise_cov_zero_for_identical_vectors() -> None:
    """Bit-identical vectors SHALL produce CoV = 0.0 (degenerate substrate)."""
    mod = _load_script_module()
    vec = torch.tensor([1.0, -2.0, 3.0, 0.5])
    assert mod.pairwise_cov(vec, vec.clone()) == pytest.approx(0.0)


def test_pairwise_cov_nonzero_for_diverse_vectors() -> None:
    """Different vectors SHALL produce a strictly positive CoV."""
    mod = _load_script_module()
    vec_a = torch.tensor([1.0, 0.0, 0.0])
    vec_b = torch.tensor([0.0, 1.0, 0.0])
    cov = mod.pairwise_cov(vec_a, vec_b)
    # ||a-b|| = sqrt(2); mean(||a||, ||b||) = 1.0 ⇒ CoV = sqrt(2).
    assert cov == pytest.approx(math.sqrt(2.0), rel=1e-6)


def test_pairwise_cov_scale_invariant() -> None:
    """Uniformly scaling both vectors SHALL leave CoV unchanged."""
    mod = _load_script_module()
    vec_a = torch.tensor([1.0, 2.0, -1.0])
    vec_b = torch.tensor([3.0, -1.0, 2.0])
    cov_base = mod.pairwise_cov(vec_a, vec_b)
    cov_scaled = mod.pairwise_cov(vec_a * 5.0, vec_b * 5.0)
    assert cov_scaled == pytest.approx(cov_base, rel=1e-6)


def test_pairwise_cov_rejects_zero_norms() -> None:
    """All-zero pairs SHALL raise (undefined CoV)."""
    mod = _load_script_module()
    zero = torch.zeros(4)
    with pytest.raises(ValueError, match="zero L2 norm"):
        mod.pairwise_cov(zero, zero)


def test_pairwise_cov_rejects_shape_mismatch() -> None:
    """Mismatched shapes SHALL raise (architecture pairing impossible)."""
    mod = _load_script_module()
    vec_a = torch.tensor([1.0, 2.0])
    vec_b = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shapes differ"):
        mod.pairwise_cov(vec_a, vec_b)


def test_compute_pairwise_cov_matrix_n_seeds_pairs() -> None:
    """N seeds SHALL produce N*(N-1)/2 triplets, sorted by (a, b)."""
    mod = _load_script_module()
    seed_vectors = {
        42: torch.tensor([1.0, 0.0, 0.0]),
        43: torch.tensor([0.0, 1.0, 0.0]),
        44: torch.tensor([0.0, 0.0, 1.0]),
    }
    triplets = mod.compute_pairwise_cov_matrix(seed_vectors)
    pairs = [(a, b) for a, b, _ in triplets]
    assert pairs == [(42, 43), (42, 44), (43, 44)]
    assert all(c > 0 for _, _, c in triplets)


def test_compute_pairwise_cov_matrix_empty_below_two_seeds() -> None:
    """Fewer than 2 seeds SHALL yield no triplets."""
    mod = _load_script_module()
    assert mod.compute_pairwise_cov_matrix({}) == []
    assert mod.compute_pairwise_cov_matrix({42: torch.tensor([1.0])}) == []


# ---------------------------------------------------------------------------
# mean_abs_bias_output (T4 magnitude surrogate)
# ---------------------------------------------------------------------------


def test_mean_abs_bias_output_nonzero_for_random_init() -> None:
    """A randomly-initialised bias-network SHALL produce non-zero mean abs output."""
    mod = _load_script_module()
    substrate = _make_substrate(seed=42)
    mag = mod.mean_abs_bias_output(substrate)
    assert mag > 0.0


def test_mean_abs_bias_output_zero_for_zero_weights() -> None:
    """Zero-weights bias-network SHALL produce near-zero mean abs output."""
    mod = _load_script_module()
    net = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 4))
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    for p in net.parameters():
        p.requires_grad = False
    substrate = TransgenerationalMemory(
        logit_bias=torch.zeros(4),
        lineage_depth=0,
        source_genome_id="zero",
        bias_network=net,
        input_features=("a", "b", "c"),
    )
    mag = mod.mean_abs_bias_output(substrate)
    assert mag == pytest.approx(0.0, abs=1e-7)


def test_mean_abs_bias_output_rejects_no_bias_network() -> None:
    """Substrate with bias_network=None SHALL raise."""
    mod = _load_script_module()
    substrate = TransgenerationalMemory(
        logit_bias=torch.zeros(4),
        lineage_depth=0,
        source_genome_id="legacy",
    )
    with pytest.raises(ValueError, match="bias_network=None"):
        mod.mean_abs_bias_output(substrate)


# ---------------------------------------------------------------------------
# evaluate_diversity end-to-end (T2 + T4 verdict assembly)
# ---------------------------------------------------------------------------


def test_evaluate_diversity_passes_for_diverse_random_seeds() -> None:
    """Four independently-seeded substrates SHALL pass T2 + T4 by default."""
    mod = _load_script_module()
    substrates = {seed: _make_substrate(seed=seed) for seed in (42, 43, 44, 45)}
    vectors = {seed: mod._flatten_state_dict(sub) for seed, sub in substrates.items()}
    verdict = mod.evaluate_diversity(vectors, substrates)
    assert verdict["diversity_pass"]
    assert verdict["magnitude_pass"]
    assert verdict["overall_pass"]
    assert verdict["n_seeds"] == 4
    assert verdict["min_pairwise_cov"] >= verdict["diversity_threshold"]


def test_evaluate_diversity_fails_when_substrates_bit_identical() -> None:
    """The M6 failure mode (bit-identical substrates) SHALL trip T2."""
    mod = _load_script_module()
    shared_net = _make_bias_network(seed=42)
    substrates = {
        seed: TransgenerationalMemory(
            logit_bias=torch.zeros(4),
            lineage_depth=0,
            source_genome_id=f"shared-{seed}",
            bias_network=shared_net,
            input_features=("a", "b", "c"),
        )
        for seed in (42, 43, 44, 45)
    }
    vectors = {seed: mod._flatten_state_dict(sub) for seed, sub in substrates.items()}
    verdict = mod.evaluate_diversity(vectors, substrates)
    assert verdict["min_pairwise_cov"] == pytest.approx(0.0, abs=1e-7)
    assert verdict["diversity_pass"] is False
    assert verdict["overall_pass"] is False


def test_evaluate_diversity_fails_when_substrates_below_threshold() -> None:
    """Near-identical substrates SHALL trip T2 with a custom threshold."""
    mod = _load_script_module()
    # Three substrates with tiny pairwise distinction → min CoV << 5%.
    substrates = {}
    for idx, seed in enumerate((42, 43, 44)):
        net = _make_bias_network(seed=42)  # start from same init
        with torch.no_grad():
            # Add tiny per-seed perturbation (idx scales the noise).
            for p_new in net.parameters():
                p_new.add_(1e-5 * (idx + 1) * torch.randn_like(p_new))
        substrates[seed] = TransgenerationalMemory(
            logit_bias=torch.zeros(4),
            lineage_depth=0,
            source_genome_id=f"perturb-{seed}",
            bias_network=net,
            input_features=("a", "b", "c"),
        )
    vectors = {seed: mod._flatten_state_dict(sub) for seed, sub in substrates.items()}
    verdict = mod.evaluate_diversity(vectors, substrates, diversity_threshold=0.05)
    assert verdict["min_pairwise_cov"] < 0.05
    assert verdict["diversity_pass"] is False


def test_evaluate_diversity_fails_on_zero_magnitude_substrate() -> None:
    """All-zero bias-networks SHALL trip T4 (substrate is the zero function)."""
    mod = _load_script_module()
    substrates = {}
    for seed in (42, 43):
        net = _make_bias_network(seed=seed)
        # Zero out only the output-layer biases + weights → mean abs out = 0.
        with torch.no_grad():
            for p in net[-1].parameters():
                p.zero_()
        substrates[seed] = TransgenerationalMemory(
            logit_bias=torch.zeros(4),
            lineage_depth=0,
            source_genome_id=f"zero-out-{seed}",
            bias_network=net,
            input_features=("a", "b", "c"),
        )
    vectors = {seed: mod._flatten_state_dict(sub) for seed, sub in substrates.items()}
    verdict = mod.evaluate_diversity(vectors, substrates)
    # Per-seed magnitudes should be 0 → T4 fails.
    assert verdict["min_magnitude"] == pytest.approx(0.0, abs=1e-7)
    assert verdict["magnitude_pass"] is False
    assert verdict["overall_pass"] is False


def test_evaluate_diversity_fails_closed_on_single_seed() -> None:
    """Single seed → no pair → diversity undefined → fail closed."""
    mod = _load_script_module()
    substrates = {42: _make_substrate(seed=42)}
    vectors = {42: mod._flatten_state_dict(substrates[42])}
    verdict = mod.evaluate_diversity(vectors, substrates)
    assert verdict["n_seeds"] == 1
    assert math.isnan(verdict["min_pairwise_cov"])
    assert verdict["diversity_pass"] is False
    assert verdict["overall_pass"] is False


def test_evaluate_diversity_fails_closed_on_empty_input() -> None:
    """Empty input → fail closed (no calibration seeds completed)."""
    mod = _load_script_module()
    verdict = mod.evaluate_diversity({}, {})
    assert verdict["n_seeds"] == 0
    assert math.isnan(verdict["min_pairwise_cov"])
    assert verdict["diversity_pass"] is False
    assert verdict["magnitude_pass"] is False
    assert verdict["overall_pass"] is False


# ---------------------------------------------------------------------------
# discover_seed_substrates + main CLI integration
# ---------------------------------------------------------------------------


def test_discover_seed_substrates_walks_inheritance_layout(tmp_path: Path) -> None:
    """Discovery SHALL walk the canonical ``arm/seed-N/.../inheritance/gen-000/`` layout."""
    mod = _load_script_module()
    arm_dir = tmp_path / "tei_on"
    for seed in (42, 43):
        substrate = _make_substrate(seed=seed)
        gen_dir = arm_dir / f"seed-{seed}" / "session-xyz" / "inheritance" / "gen-000"
        substrate_path = gen_dir / f"genome-test-{seed}.tei.pt"
        save_substrate(substrate, substrate_path)
    paths = mod.discover_seed_substrates(tmp_path, arm="tei_on")
    assert set(paths.keys()) == {42, 43}
    assert all(p.exists() for p in paths.values())


def test_discover_seed_substrates_returns_empty_for_missing_arm(tmp_path: Path) -> None:
    """Missing arm directory SHALL return empty dict (caller decides)."""
    mod = _load_script_module()
    paths = mod.discover_seed_substrates(tmp_path, arm="tei_on")
    assert paths == {}


def test_main_cli_exit_1_on_no_substrates(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """CLI SHALL exit 1 when no substrates are discovered (calibration smoke incomplete)."""
    mod = _load_script_module()
    code = mod.main(["--campaign-root", str(tmp_path), "--arm", "tei_on"])
    assert code == 1
    out = capsys.readouterr().out
    assert "overall_pass" in out


def test_main_cli_exit_0_on_pass(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """CLI SHALL exit 0 + emit JSON ``overall_pass=true`` when T2 + T4 hold."""
    mod = _load_script_module()
    arm_dir = tmp_path / "tei_on"
    for seed in (42, 43, 44, 45):
        substrate = _make_substrate(seed=seed)
        gen_dir = arm_dir / f"seed-{seed}" / "session-xyz" / "inheritance" / "gen-000"
        substrate_path = gen_dir / f"genome-test-{seed}.tei.pt"
        save_substrate(substrate, substrate_path)
    csv_out = tmp_path / "diversity.csv"
    code = mod.main(
        [
            "--campaign-root",
            str(tmp_path),
            "--arm",
            "tei_on",
            "--output-csv",
            str(csv_out),
        ],
    )
    assert code == 0
    captured = capsys.readouterr()
    assert '"overall_pass": true' in captured.out
    assert csv_out.exists()
    csv_text = csv_out.read_text(encoding="utf-8")
    assert "pairwise_cov" in csv_text
    assert "mean_abs_bias_output" in csv_text
    assert "overall_pass" in csv_text
