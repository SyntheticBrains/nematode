# pragma: no cover

"""QRH Mutual Information Decision Gate.

Compares the mutual information (MI) between quantum reservoir features
and oracle-labelled optimal actions for three feature extraction methods:

1. **Structured** QRH reservoir (C. elegans connectome topology)
2. **Random** QRH reservoir (same gate density, random topology)
3. **Classical** MLP hidden features (no quantum processing)

The structured reservoir must achieve significantly higher MI than the
random reservoir (permutation test, p < 0.01) to justify the C. elegans
topology hypothesis.

Usage
-----
    uv run scripts/qrh_mi_analysis.py [OPTIONS]

Examples
--------
    # Default analysis (1000 samples, 1000 permutations)
    uv run scripts/qrh_mi_analysis.py

    # Quick test run
    uv run scripts/qrh_mi_analysis.py --num-samples 200 --num-permutations 100

    # Custom seed for reproducibility
    uv run scripts/qrh_mi_analysis.py --seed 123
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Oracle and brain imports
from quantumnematode.brain.arch import BrainParams, MLPPPOBrainConfig
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain
from quantumnematode.brain.arch.qrh import (
    QRHBrain,
    QRHBrainConfig,
    _compute_feature_dim,
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
DEFAULT_RESERVOIR_DEPTH = 3
DEFAULT_OUTPUT_DIR = "exports/mi_analysis"

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

    structured_mi: MIResult | None = None
    random_mi: MIResult | None = None
    classical_mi: MIResult | None = None
    permutation_test: PermutationTestResult | None = None
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
    to the best discrete action (FORWARD/LEFT/RIGHT/STAY). This provides
    the "ground truth" that a good feature representation should be able to
    predict — i.e., features that carry high MI with these labels encode
    task-relevant information about optimal navigation.

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
    # Action indices: 0=FORWARD, 1=LEFT, 2=RIGHT, 3=STAY
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
            # No signal: stay in place
            labels[i] = 3  # STAY
        elif abs(relative_angle) < np.pi / 4:
            # Gradient ahead: go forward
            labels[i] = 0  # FORWARD
        elif relative_angle > 0:
            # Gradient to the left: turn left
            labels[i] = 1  # LEFT
        else:
            # Gradient to the right: turn right
            labels[i] = 2  # RIGHT

    return labels


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_structured_features(
    params_list: list[BrainParams],
    num_qubits: int,
    reservoir_depth: int,
    seed: int,
) -> np.ndarray:
    """Extract Z/ZZ features from structured (C. elegans) reservoir.

    Parameters
    ----------
    params_list : list[BrainParams]
        Input observations.
    num_qubits : int
        Number of reservoir qubits.
    reservoir_depth : int
        Number of reservoir layers.
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
        reservoir_depth=reservoir_depth,
        use_random_topology=False,
    )
    brain = QRHBrain(config=config)

    feature_dim = _compute_feature_dim(num_qubits)
    features = np.zeros((len(params_list), feature_dim), dtype=np.float32)

    for i, params in enumerate(params_list):
        sensory = brain.preprocess(params)
        features[i] = brain._get_reservoir_features(sensory)  # noqa: SLF001

    return features


def extract_random_features(
    params_list: list[BrainParams],
    num_qubits: int,
    reservoir_depth: int,
    seed: int,
) -> np.ndarray:
    """Extract Z/ZZ features from random topology reservoir.

    Parameters
    ----------
    params_list : list[BrainParams]
        Input observations.
    num_qubits : int
        Number of reservoir qubits.
    reservoir_depth : int
        Number of reservoir layers.
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
        reservoir_depth=reservoir_depth,
        use_random_topology=True,
    )
    brain = QRHBrain(config=config)

    feature_dim = _compute_feature_dim(num_qubits)
    features = np.zeros((len(params_list), feature_dim), dtype=np.float32)

    for i, params in enumerate(params_list):
        sensory = brain.preprocess(params)
        features[i] = brain._get_reservoir_features(sensory)  # noqa: SLF001

    return features


def extract_classical_features(
    params_list: list[BrainParams],
    seed: int,
) -> np.ndarray:
    """Extract hidden-layer features from a classical MLP.

    Uses the MLPPPO actor's penultimate hidden layer activations
    as the classical feature baseline.

    Parameters
    ----------
    params_list : list[BrainParams]
        Input observations.
    seed : int
        Seed for the MLP brain.

    Returns
    -------
    np.ndarray
        Feature matrix, shape (num_samples, hidden_dim).
    """
    config = MLPPPOBrainConfig(
        seed=seed,
        actor_hidden_dim=64,
        critic_hidden_dim=64,
        num_hidden_layers=2,
    )
    brain = MLPPPOBrain(config=config)

    # Extract features from the penultimate hidden layer
    # The actor is a Sequential: Linear -> ReLU -> Linear -> ReLU -> Linear(output)
    # We want the output after the last ReLU (before the final linear)
    hidden_dim = config.actor_hidden_dim
    features = np.zeros((len(params_list), hidden_dim), dtype=np.float32)

    # Hook into the actor network to get hidden activations
    activation = {}

    def get_activation(name: str):  # noqa: ANN202
        def hook(module, input, output) -> None:  # noqa: A002, ARG001, ANN001
            activation[name] = output.detach().cpu().numpy()

        return hook

    # Find the last ReLU in the actor network
    actor_modules = list(brain.actor.modules())
    relu_indices = [i for i, m in enumerate(actor_modules) if isinstance(m, torch.nn.ReLU)]
    if relu_indices:
        last_relu = actor_modules[relu_indices[-1]]
        last_relu.register_forward_hook(get_activation("hidden"))

    for i, params in enumerate(params_list):
        x = brain.preprocess(params)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        brain.forward_actor(x_tensor)
        if "hidden" in activation:
            features[i] = activation["hidden"].flatten()

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


def permutation_test(
    features_a: np.ndarray,
    features_b: np.ndarray,
    labels: np.ndarray,
    num_permutations: int,
    seed: int,
) -> PermutationTestResult:
    """Run a permutation test comparing mean MI of two feature sets.

    Tests H0: mean MI(structured) = mean MI(random) via row-swap permutation.
    For each permutation, each sample's feature row is randomly assigned to
    group A or group B (regardless of its original source), and the MI delta
    is recomputed. The p-value is the fraction of permuted deltas >= observed.

    Parameters
    ----------
    features_a : np.ndarray
        Feature matrix for method A (structured), shape (n, d).
    features_b : np.ndarray
        Feature matrix for method B (random), shape (n, d).
    labels : np.ndarray
        True action labels, shape (n,).
    num_permutations : int
        Number of permutations.
    seed : int
        RNG seed.

    Returns
    -------
    PermutationTestResult
        Observed delta, p-value, and significance.
    """
    n = len(labels)

    # Observed MI difference
    mi_a = mutual_info_classif(features_a, labels, discrete_features="auto", random_state=0)
    mi_b = mutual_info_classif(features_b, labels, discrete_features="auto", random_state=0)
    observed_delta = float(np.mean(mi_a) - np.mean(mi_b))

    rng = np.random.default_rng(seed)
    count_ge = 0

    for _ in range(num_permutations):
        # Randomly assign each sample to group A or B
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
        method_a="structured",
        method_b="random",
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
    print("QRH Mutual Information Decision Gate — Results")
    print("=" * 70)
    print(f"  Samples:       {results.num_samples}")
    print(f"  Qubits:        {results.num_qubits}")
    print(f"  Seed:          {results.seed}")
    print(f"  Elapsed:       {results.elapsed_seconds:.1f}s")
    print()

    print("Feature Extraction Methods:")
    print("-" * 70)
    for mi_result, label in [
        (results.structured_mi, "Structured (C. elegans)"),
        (results.random_mi, "Random topology"),
        (results.classical_mi, "Classical MLP hidden"),
    ]:
        if mi_result is not None:
            print(
                f"  {label:30s}  mean MI = {mi_result.mean_mi:.4f}  "
                f"total MI = {mi_result.total_mi:.4f}",
            )
    print()

    if results.permutation_test is not None:
        pt = results.permutation_test
        print("Permutation Test (Structured vs Random):")
        print("-" * 70)
        print(f"  Observed delta (mean MI):  {pt.observed_delta:+.4f}")
        print(f"  p-value:                   {pt.p_value:.4f}")
        print(f"  Permutations:              {pt.num_permutations}")
        sig_str = "YES" if pt.significant else "NO"
        print(f"  Significant (p < {DEFAULT_SIGNIFICANCE_LEVEL}):    {sig_str}")
        print()

    # Decision
    print("=" * 70)
    if results.permutation_test is not None and results.permutation_test.significant:
        print("DECISION: GO")
        print("  Structured topology provides significantly higher MI than random.")
        print("  Proceed with QRH training experiments.")
    else:
        print("DECISION: NO-GO")
        print("  Structured topology does NOT provide significantly higher MI.")
        print("  Review topology design before proceeding.")
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

    data = {
        "num_samples": results.num_samples,
        "num_qubits": results.num_qubits,
        "seed": results.seed,
        "elapsed_seconds": results.elapsed_seconds,
        "structured_mi": {
            "mean_mi": results.structured_mi.mean_mi,
            "total_mi": results.structured_mi.total_mi,
            "mi_per_feature": results.structured_mi.mi_per_feature,
        }
        if results.structured_mi
        else None,
        "random_mi": {
            "mean_mi": results.random_mi.mean_mi,
            "total_mi": results.random_mi.total_mi,
            "mi_per_feature": results.random_mi.mi_per_feature,
        }
        if results.random_mi
        else None,
        "classical_mi": {
            "mean_mi": results.classical_mi.mean_mi,
            "total_mi": results.classical_mi.total_mi,
            "mi_per_feature": results.classical_mi.mi_per_feature,
        }
        if results.classical_mi
        else None,
        "permutation_test": {
            "observed_delta": results.permutation_test.observed_delta,
            "p_value": results.permutation_test.p_value,
            "significant": results.permutation_test.significant,
            "num_permutations": results.permutation_test.num_permutations,
        }
        if results.permutation_test
        else None,
        "decision": "GO"
        if (results.permutation_test and results.permutation_test.significant)
        else "NO-GO",
    }

    filepath = output_path / "mi_analysis_results.json"
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
        description="QRH Mutual Information Decision Gate: "
        "Compare structured vs random quantum reservoir feature quality.",
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
        help="Number of reservoir qubits.",
    )
    parser.add_argument(
        "--reservoir-depth",
        type=int,
        default=DEFAULT_RESERVOIR_DEPTH,
        help="Number of reservoir entanglement layers.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saving JSON results.",
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
    """Run the MI decision gate analysis."""
    args = parse_arguments()
    start_time = time.time()

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    results = AnalysisResults(
        num_samples=args.num_samples,
        num_qubits=args.num_qubits,
        seed=args.seed,
    )

    # Step 1: Generate synthetic dataset
    print(f"Generating {args.num_samples} synthetic observations...")
    params_list = generate_synthetic_brain_params(args.num_samples, rng)

    # Step 2: Generate oracle labels using rule-based gradient-following policy
    print("Generating oracle action labels (gradient-following policy)...")
    labels = generate_oracle_labels(params_list, seed=args.seed)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Label distribution: {dict(zip(unique, counts, strict=True))}")

    # Step 3: Extract features from all three methods
    print(
        f"Extracting structured reservoir features ({args.num_qubits}q, "
        f"depth={args.reservoir_depth})...",
    )
    structured_features = extract_structured_features(
        params_list,
        args.num_qubits,
        args.reservoir_depth,
        args.seed,
    )
    print(f"  Shape: {structured_features.shape}")

    print(
        f"Extracting random reservoir features ({args.num_qubits}q, "
        f"depth={args.reservoir_depth})...",
    )
    random_features = extract_random_features(
        params_list,
        args.num_qubits,
        args.reservoir_depth,
        args.seed,
    )
    print(f"  Shape: {random_features.shape}")

    print("Extracting classical MLP hidden features...")
    classical_features = extract_classical_features(params_list, seed=args.seed)
    print(f"  Shape: {classical_features.shape}")

    # Step 4: Compute MI for each method
    print("Computing mutual information...")
    structured_mi = compute_mi(structured_features, labels)
    structured_mi.method = "structured"
    results.structured_mi = structured_mi

    random_mi = compute_mi(random_features, labels)
    random_mi.method = "random"
    results.random_mi = random_mi

    classical_mi = compute_mi(classical_features, labels)
    classical_mi.method = "classical"
    results.classical_mi = classical_mi

    print(f"  Structured mean MI: {structured_mi.mean_mi:.4f}")
    print(f"  Random mean MI:     {random_mi.mean_mi:.4f}")
    print(f"  Classical mean MI:  {classical_mi.mean_mi:.4f}")

    # Step 5: Permutation test
    if not args.skip_permutation:
        print(f"Running permutation test ({args.num_permutations} permutations)...")
        perm_result = permutation_test(
            structured_features,
            random_features,
            labels,
            num_permutations=args.num_permutations,
            seed=args.seed,
        )
        results.permutation_test = perm_result
    else:
        print("Skipping permutation test (--skip-permutation).")

    results.elapsed_seconds = time.time() - start_time

    # Output results
    print_results(results)
    save_results(results, args.output_dir)

    # Exit code: 0 = GO (significant), 1 = NO-GO
    if results.permutation_test is not None and results.permutation_test.significant:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
