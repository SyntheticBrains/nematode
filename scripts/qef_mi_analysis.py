# pragma: no cover

"""QEF Mutual Information Decision Gate.

Compares the mutual information (MI) between quantum features
and oracle-labelled optimal actions for three feature extraction methods:

1. **Entangled** QEF circuit (configurable topology, default modality-paired)
2. **Separable** QEF circuit (same circuit but no CZ gates)
3. **QRH Random** reservoir (random topology baseline from QRH)

The entangled features must achieve higher MI than both separable and QRH
random features. Decision criteria from the research doc (Week 1 gate):
- If entangled MI <= separable MI → try alternative topologies
- If entangled MI <= QRH random MI → stop (no advantage over random reservoir)

Usage
-----
    uv run scripts/qef_mi_analysis.py [OPTIONS]

Examples
--------
    # Default analysis (1000 samples, 1000 permutations)
    uv run scripts/qef_mi_analysis.py

    # Quick test run
    uv run scripts/qef_mi_analysis.py --num-samples 200 --num-permutations 100

    # Test ring topology
    uv run scripts/qef_mi_analysis.py --topology ring

    # Skip permutation test for faster development iteration
    uv run scripts/qef_mi_analysis.py --skip-permutation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.qef import (
    QEFBrain,
    QEFBrainConfig,
)
from quantumnematode.brain.arch.qef import (
    _compute_feature_dim as qef_compute_feature_dim,
)
from quantumnematode.brain.arch.qrh import (
    QRHBrain,
    QRHBrainConfig,
)
from quantumnematode.brain.arch.qrh import (
    _compute_feature_dim as qrh_compute_feature_dim,
)
from quantumnematode.env import Direction
from quantumnematode.utils.seeding import set_global_seed
from sklearn.feature_selection import mutual_info_classif

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NUM_SAMPLES = 1000
DEFAULT_NUM_PERMUTATIONS = 1000
DEFAULT_SEED = 42
DEFAULT_SIGNIFICANCE_LEVEL = 0.01
DEFAULT_NUM_QUBITS = 8
DEFAULT_CIRCUIT_DEPTH = 2
DEFAULT_OUTPUT_DIR = "exports/qef_mi_analysis"

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class MIResult:
    """Result of MI computation for a single feature extraction method."""

    method: str
    mi_per_feature: list[float] = field(default_factory=list)
    mean_mi: float = 0.0
    total_mi: float = 0.0


@dataclass
class PermutationTestResult:
    """Result of a permutation test comparing two methods."""

    method_a: str
    method_b: str
    observed_delta: float
    p_value: float
    significant: bool
    num_permutations: int


@dataclass
class AnalysisResults:
    """Complete analysis results."""

    entangled_mi: MIResult | None = None
    separable_mi: MIResult | None = None
    qrh_random_mi: MIResult | None = None
    perm_test_vs_separable: PermutationTestResult | None = None
    perm_test_vs_qrh: PermutationTestResult | None = None
    topology: str = "modality_paired"
    num_samples: int = 0
    num_qubits: int = 0
    seed: int = 0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Dataset Generation
# ---------------------------------------------------------------------------


def generate_synthetic_brain_params(
    num_samples: int,
    rng: np.random.Generator,
) -> list[BrainParams]:
    """Generate synthetic BrainParams with diverse sensory inputs.

    Creates a dataset of environment observations by sampling gradient
    strengths, directions, and agent facing directions uniformly.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    rng : numpy.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    list[BrainParams]
        Synthetic sensory observations.
    """
    params_list = []
    for _ in range(num_samples):
        grad_strength = float(rng.uniform(0.0, 1.0))
        grad_direction = float(rng.uniform(-np.pi, np.pi))
        agent_dir = DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))]

        params_list.append(
            BrainParams(
                gradient_strength=grad_strength,
                gradient_direction=grad_direction,
                agent_direction=agent_dir,
            ),
        )
    return params_list


def generate_oracle_labels(
    params_list: list[BrainParams],
    seed: int,  # noqa: ARG001
) -> np.ndarray:
    """Generate oracle action labels from a deterministic gradient-following policy.

    Implements a rule-based oracle that maps the relative gradient direction
    to the best discrete action (FORWARD/LEFT/RIGHT/STAY).

    Parameters
    ----------
    params_list : list[BrainParams]
        Observations to label.
    seed : int
        Seed (reserved for future stochastic oracles).

    Returns
    -------
    np.ndarray
        Integer action labels, shape (num_samples,).
    """
    labels = np.zeros(len(params_list), dtype=np.int64)

    for i, params in enumerate(params_list):
        grad_strength = float(params.gradient_strength or 0.0)
        grad_direction = float(params.gradient_direction or 0.0)
        direction_map = {
            Direction.UP: np.pi / 2,
            Direction.DOWN: -np.pi / 2,
            Direction.LEFT: np.pi,
            Direction.RIGHT: 0.0,
        }
        agent_dir = params.agent_direction or Direction.UP
        agent_angle = direction_map.get(agent_dir, np.pi / 2)
        relative_angle = (grad_direction - agent_angle + np.pi) % (2 * np.pi) - np.pi

        if grad_strength < 0.1:
            labels[i] = 3  # STAY
        elif abs(relative_angle) < np.pi / 4:
            labels[i] = 0  # FORWARD
        elif relative_angle > 0:
            labels[i] = 1  # LEFT
        else:
            labels[i] = 2  # RIGHT

    return labels


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_qef_features(  # noqa: PLR0913
    params_list: list[BrainParams],
    num_qubits: int,
    circuit_depth: int,
    seed: int,
    topology: Literal["modality_paired", "ring", "random"],
    *,
    entanglement_enabled: bool,
) -> np.ndarray:
    """Extract Z + ZZ + cos/sin features from a QEF circuit.

    Parameters
    ----------
    params_list : list[BrainParams]
        Input observations.
    num_qubits : int
        Number of circuit qubits.
    circuit_depth : int
        Number of encoding + entanglement layers.
    seed : int
        Circuit seed.
    topology : str
        Entanglement topology ("modality_paired", "ring", "random").
    entanglement_enabled : bool
        Whether to apply CZ entanglement gates.

    Returns
    -------
    np.ndarray
        Feature matrix, shape (num_samples, feature_dim).
    """
    config = QEFBrainConfig(
        seed=seed,
        num_qubits=num_qubits,
        circuit_depth=circuit_depth,
        circuit_seed=seed,
        entanglement_topology=topology,
        entanglement_enabled=entanglement_enabled,
    )
    brain = QEFBrain(config=config)

    feature_dim = qef_compute_feature_dim(num_qubits)
    features = np.zeros((len(params_list), feature_dim), dtype=np.float32)

    for i, params in enumerate(params_list):
        sensory = brain.preprocess(params)
        features[i] = brain._get_reservoir_features(sensory)  # noqa: SLF001

    return features


def extract_qrh_random_features(
    params_list: list[BrainParams],
    num_qubits: int,
    seed: int,
) -> np.ndarray:
    """Extract X/Y/Z + ZZ features from QRH random topology reservoir.

    Parameters
    ----------
    params_list : list[BrainParams]
        Input observations.
    num_qubits : int
        Number of reservoir qubits.
    seed : int
        Reservoir seed.

    Returns
    -------
    np.ndarray
        Feature matrix, shape (num_samples, feature_dim).
    """
    config = QRHBrainConfig(
        seed=seed,
        num_reservoir_qubits=num_qubits,
        reservoir_depth=3,
        use_random_topology=True,
    )
    brain = QRHBrain(config=config)

    feature_dim = qrh_compute_feature_dim(num_qubits)
    features = np.zeros((len(params_list), feature_dim), dtype=np.float32)

    for i, params in enumerate(params_list):
        sensory = brain.preprocess(params)
        features[i] = brain._get_reservoir_features(sensory)  # noqa: SLF001

    return features


# ---------------------------------------------------------------------------
# MI Computation
# ---------------------------------------------------------------------------


def compute_mi(features: np.ndarray, labels: np.ndarray) -> MIResult:
    """Compute mutual information between features and discrete labels.

    Uses sklearn's mutual_info_classif which estimates MI using k-nearest
    neighbors for continuous features.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix, shape (num_samples, feature_dim).
    labels : np.ndarray
        Integer class labels, shape (num_samples,).

    Returns
    -------
    MIResult
        Per-feature MI values and aggregates.
    """
    mi_values = mutual_info_classif(
        features,
        labels,
        discrete_features="auto",
        random_state=0,
    )

    return MIResult(
        method="",
        mi_per_feature=mi_values.tolist(),
        mean_mi=float(np.mean(mi_values)),
        total_mi=float(np.sum(mi_values)),
    )


# ---------------------------------------------------------------------------
# Permutation Test
# ---------------------------------------------------------------------------


def permutation_test(  # noqa: PLR0913
    features_a: np.ndarray,
    features_b: np.ndarray,
    labels: np.ndarray,
    num_permutations: int,
    seed: int,
    method_a: str = "entangled",
    method_b: str = "baseline",
) -> PermutationTestResult:
    """Run a permutation test comparing mean MI of two feature sets.

    Tests H0: mean MI(A) = mean MI(B) via row-swap permutation.
    For each permutation, each sample's feature row is randomly assigned to
    group A or group B, and the MI delta is recomputed.

    Parameters
    ----------
    features_a : np.ndarray
        Feature matrix for method A, shape (n, d).
    features_b : np.ndarray
        Feature matrix for method B, shape (n, d).
    labels : np.ndarray
        True action labels, shape (n,).
    num_permutations : int
        Number of permutations.
    seed : int
        RNG seed.
    method_a : str
        Name of method A for reporting.
    method_b : str
        Name of method B for reporting.

    Returns
    -------
    PermutationTestResult
        Observed delta, p-value, and significance.
    """
    n = len(labels)

    mi_a = mutual_info_classif(features_a, labels, discrete_features="auto", random_state=0)
    mi_b = mutual_info_classif(features_b, labels, discrete_features="auto", random_state=0)
    observed_delta = float(np.mean(mi_a) - np.mean(mi_b))

    rng = np.random.default_rng(seed)
    count_ge = 0

    for _ in range(num_permutations):
        swap_mask = rng.random(n) < 0.5
        perm_a = np.where(swap_mask[:, None], features_b, features_a)
        perm_b = np.where(swap_mask[:, None], features_a, features_b)

        perm_mi_a = mutual_info_classif(
            perm_a,
            labels,
            discrete_features="auto",
            random_state=0,
        )
        perm_mi_b = mutual_info_classif(
            perm_b,
            labels,
            discrete_features="auto",
            random_state=0,
        )
        perm_delta = float(np.mean(perm_mi_a) - np.mean(perm_mi_b))

        if perm_delta >= observed_delta:
            count_ge += 1

    p_value = (count_ge + 1) / (num_permutations + 1)

    return PermutationTestResult(
        method_a=method_a,
        method_b=method_b,
        observed_delta=observed_delta,
        p_value=p_value,
        significant=p_value < DEFAULT_SIGNIFICANCE_LEVEL,
        num_permutations=num_permutations,
    )


# ---------------------------------------------------------------------------
# Results Output
# ---------------------------------------------------------------------------


def print_results(results: AnalysisResults) -> None:
    """Print analysis results in a structured format."""
    print("\n" + "=" * 70)
    print("QEF Mutual Information Decision Gate — Results")
    print("=" * 70)
    print(f"  Samples:       {results.num_samples}")
    print(f"  Qubits:        {results.num_qubits}")
    print(f"  Topology:      {results.topology}")
    print(f"  Seed:          {results.seed}")
    print(f"  Elapsed:       {results.elapsed_seconds:.1f}s")
    print()

    print("Feature Extraction Methods:")
    print("-" * 70)
    for mi_result, label in [
        (results.entangled_mi, f"QEF Entangled ({results.topology})"),
        (results.separable_mi, "QEF Separable (no CZ)"),
        (results.qrh_random_mi, "QRH Random reservoir"),
    ]:
        if mi_result is not None:
            print(
                f"  {label:35s}  mean MI = {mi_result.mean_mi:.4f}  "
                f"total MI = {mi_result.total_mi:.4f}",
            )
    print()

    for perm_result, label in [
        (results.perm_test_vs_separable, "Entangled vs Separable"),
        (results.perm_test_vs_qrh, "Entangled vs QRH Random"),
    ]:
        if perm_result is not None:
            print(f"Permutation Test ({label}):")
            print("-" * 70)
            print(f"  Observed delta (mean MI):  {perm_result.observed_delta:+.4f}")
            print(f"  p-value:                   {perm_result.p_value:.4f}")
            print(f"  Permutations:              {perm_result.num_permutations}")
            sig_str = "YES" if perm_result.significant else "NO"
            print(f"  Significant (p < {DEFAULT_SIGNIFICANCE_LEVEL}):    {sig_str}")
            print()

    # Decision
    print("=" * 70)
    vs_sep = results.perm_test_vs_separable
    vs_qrh = results.perm_test_vs_qrh

    if vs_sep and not vs_sep.significant:
        print("DECISION: TRY ALTERNATIVE TOPOLOGY")
        print("  Entangled MI is NOT significantly higher than separable MI.")
        print("  The entanglement topology does not add value. Try ring or random.")
    elif vs_qrh and not vs_qrh.significant:
        print("DECISION: STOP")
        print("  Entangled MI is NOT significantly higher than QRH random MI.")
        print("  Purposeful entanglement provides no advantage over random reservoir.")
    elif vs_sep and vs_sep.significant and vs_qrh and vs_qrh.significant:
        print("DECISION: GO")
        print("  Entangled features provide significantly higher MI than both")
        print("  separable and QRH random baselines. Proceed with QEF training.")
    else:
        print("DECISION: INCOMPLETE")
        print("  Permutation tests were not run. Use without --skip-permutation")
        print("  for a full decision.")
    print("=" * 70)


def save_results(results: AnalysisResults, output_dir: str) -> Path:
    """Save results as JSON for programmatic consumption.

    Parameters
    ----------
    results : AnalysisResults
        Complete analysis results.
    output_dir : str
        Directory to save results in.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def _mi_dict(mi: MIResult | None) -> dict | None:
        if mi is None:
            return None
        return {
            "method": mi.method,
            "mean_mi": mi.mean_mi,
            "total_mi": mi.total_mi,
            "mi_per_feature": mi.mi_per_feature,
        }

    def _perm_dict(pt: PermutationTestResult | None) -> dict | None:
        if pt is None:
            return None
        return {
            "method_a": pt.method_a,
            "method_b": pt.method_b,
            "observed_delta": pt.observed_delta,
            "p_value": pt.p_value,
            "significant": pt.significant,
            "num_permutations": pt.num_permutations,
        }

    data = {
        "num_samples": results.num_samples,
        "num_qubits": results.num_qubits,
        "topology": results.topology,
        "seed": results.seed,
        "elapsed_seconds": results.elapsed_seconds,
        "entangled_mi": _mi_dict(results.entangled_mi),
        "separable_mi": _mi_dict(results.separable_mi),
        "qrh_random_mi": _mi_dict(results.qrh_random_mi),
        "perm_test_vs_separable": _perm_dict(results.perm_test_vs_separable),
        "perm_test_vs_qrh": _perm_dict(results.perm_test_vs_qrh),
    }

    filepath = output_path / "qef_mi_analysis_results.json"
    with filepath.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="QEF Mutual Information Decision Gate: "
        "Compare entangled vs separable vs QRH random feature quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of synthetic observations to generate.",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=DEFAULT_NUM_PERMUTATIONS,
        help="Number of permutations for significance testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=DEFAULT_NUM_QUBITS,
        help="Number of circuit qubits.",
    )
    parser.add_argument(
        "--circuit-depth",
        type=int,
        default=DEFAULT_CIRCUIT_DEPTH,
        help="Number of encoding + entanglement layers.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="modality_paired",
        choices=["modality_paired", "ring", "random"],
        help="Entanglement topology to test.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saving JSON results.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="both",
        choices=["text", "json", "both"],
        help="Output format: text (console), json (file), or both.",
    )
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Skip the permutation test (faster, for development).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the QEF MI decision gate analysis."""
    args = parse_arguments()
    start_time = time.time()

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    results = AnalysisResults(
        topology=args.topology,
        num_samples=args.num_samples,
        num_qubits=args.num_qubits,
        seed=args.seed,
    )

    # Step 1: Generate synthetic dataset
    print(f"Generating {args.num_samples} synthetic observations...")
    params_list = generate_synthetic_brain_params(args.num_samples, rng)

    # Step 2: Generate oracle labels
    print("Generating oracle action labels (gradient-following policy)...")
    labels = generate_oracle_labels(params_list, seed=args.seed)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Label distribution: {dict(zip(unique, counts, strict=True))}")

    # Step 3: Extract features from all three methods
    print(
        f"Extracting QEF entangled features ({args.topology}, "
        f"{args.num_qubits}q, depth={args.circuit_depth})...",
    )
    entangled_features = extract_qef_features(
        params_list,
        args.num_qubits,
        args.circuit_depth,
        args.seed,
        topology=args.topology,
        entanglement_enabled=True,
    )
    print(f"  Shape: {entangled_features.shape}")

    print(
        f"Extracting QEF separable features ({args.num_qubits}q, depth={args.circuit_depth})...",
    )
    separable_features = extract_qef_features(
        params_list,
        args.num_qubits,
        args.circuit_depth,
        args.seed,
        topology=args.topology,
        entanglement_enabled=False,
    )
    print(f"  Shape: {separable_features.shape}")

    print(f"Extracting QRH random reservoir features ({args.num_qubits}q)...")
    qrh_features = extract_qrh_random_features(
        params_list,
        args.num_qubits,
        args.seed,
    )
    print(f"  Shape: {qrh_features.shape}")

    # Step 4: Compute MI for each method
    print("Computing mutual information...")
    entangled_mi = compute_mi(entangled_features, labels)
    entangled_mi.method = f"entangled_{args.topology}"
    results.entangled_mi = entangled_mi

    separable_mi = compute_mi(separable_features, labels)
    separable_mi.method = "separable"
    results.separable_mi = separable_mi

    qrh_mi = compute_mi(qrh_features, labels)
    qrh_mi.method = "qrh_random"
    results.qrh_random_mi = qrh_mi

    print(f"  QEF Entangled mean MI:  {entangled_mi.mean_mi:.4f}")
    print(f"  QEF Separable mean MI:  {separable_mi.mean_mi:.4f}")
    print(f"  QRH Random mean MI:     {qrh_mi.mean_mi:.4f}")

    # Step 5: Permutation tests
    if not args.skip_permutation:
        print(f"Running permutation tests ({args.num_permutations} permutations)...")

        print("  Testing: entangled vs separable...")
        results.perm_test_vs_separable = permutation_test(
            entangled_features,
            separable_features,
            labels,
            num_permutations=args.num_permutations,
            seed=args.seed,
            method_a=f"entangled_{args.topology}",
            method_b="separable",
        )

        print("  Testing: entangled vs QRH random...")
        results.perm_test_vs_qrh = permutation_test(
            entangled_features,
            qrh_features,
            labels,
            num_permutations=args.num_permutations,
            seed=args.seed + 1,  # Different seed for independence
            method_a=f"entangled_{args.topology}",
            method_b="qrh_random",
        )
    else:
        print("Skipping permutation tests (--skip-permutation).")

    results.elapsed_seconds = time.time() - start_time

    # Output results
    if args.output_format in ("text", "both"):
        print_results(results)
    if args.output_format in ("json", "both"):
        save_results(results, args.output_dir)

    # Exit code based on decision
    vs_sep = results.perm_test_vs_separable
    vs_qrh = results.perm_test_vs_qrh

    if vs_sep and vs_sep.significant and vs_qrh and vs_qrh.significant:
        return 0  # GO
    return 1  # NO-GO or incomplete


if __name__ == "__main__":
    sys.exit(main())
