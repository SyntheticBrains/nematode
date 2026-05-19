r"""Substrate-diversity tripwire (T2) for the M6.9+ TEI re-evaluation.

Computes the pairwise coefficient-of-variation (CoV) across the F0
calibration seeds' extracted ``bias_network.state_dict()`` tensors and
emits a pass/fail flag against a configurable threshold (default 5%).

This is the **second tripwire** in the pre-flight calibration smoke
gate. It catches the failure mode that closed the prior TEI evaluation
INCONCLUSIVE: 3-of-4 calibration seeds extracting bit-identical
substrates from independent F0 elites (the env+reward attractor
collapse). If pairwise CoV is below the threshold, the substrate is
degenerate-by-construction and pilot compute would be wasted — STOP
and retune ``bias_network.hidden_dim``, ``input_features``, or the
env+reward shape before re-running calibration smoke.

**Pairwise CoV definition**:

    For each pair (i, j) of seeds, flatten + concatenate each seed's
    ``bias_network.state_dict()`` tensors into a single 1-D vector.
    Pairwise CoV(i, j) = ||W_i - W_j|| / ((||W_i|| + ||W_j||) / 2)
    where ||.|| is the L2 norm. The script reports the **minimum**
    pairwise CoV across all pairs (the worst-case collapse signature)
    and fails the threshold if min < threshold.

A second secondary metric — **mean absolute bias output** over a
deterministic probe set — is also surfaced for the T4 substrate-
magnitude tripwire. Substrate is considered degenerate at the output
level if mean ``|bias_network(probe_inputs)|`` < 0.1 (default).

Usage:
  scripts/campaigns/m69_substrate_diversity.py \\
      --campaign-root evolution_results/m69_transgenerational \\
      --arm tei_on \\
      --output-csv evaluations/m69_transgenerational/substrate_diversity.csv \\
      --threshold 0.05 \\
      --magnitude-threshold 0.1

Returns exit code 0 iff (min pairwise CoV >= threshold AND mean abs
bias output >= magnitude threshold). Exit code 1 on tripwire failure
(printed reason on stderr + JSON summary on stdout).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import torch
from quantumnematode.agent.transgenerational_memory import (
    TransgenerationalMemory,
)
from quantumnematode.agent.transgenerational_memory import (
    load as load_substrate,
)

logger = logging.getLogger(__name__)


# Defaults match plan v2 § D5 calibration-smoke tripwire envelope.
DEFAULT_DIVERSITY_THRESHOLD = 0.05  # 5% pairwise CoV (T2)
DEFAULT_MAGNITUDE_THRESHOLD = 0.1  # mean |bias output| (T4)


def _flatten_state_dict(substrate: TransgenerationalMemory) -> torch.Tensor:
    """Flatten + concatenate every parameter tensor in ``bias_network.state_dict()``.

    Returns a 1-D float tensor on CPU suitable for pairwise norm
    arithmetic. Raises ``ValueError`` if the substrate has no
    ``bias_network`` (the diversity check is only meaningful for the
    sensory-conditional bias-network form).

    Parameter ordering is determined by ``state_dict()``'s
    insertion-order iteration; PyTorch guarantees the same key order
    across calls for identically-constructed modules, so two seeds'
    flattened vectors are dimensionally comparable iff their
    architectures match.
    """
    if substrate.bias_network is None:
        msg = (
            "Substrate diversity is only defined for the sensory-conditional "
            "bias-network form; this substrate has bias_network=None "
            "(legacy logit-bias path). Re-extract with a configured "
            "bias_network in the YAML."
        )
        raise ValueError(msg)
    parts = [
        t.detach().to(dtype=torch.float64, device="cpu").flatten()
        for t in substrate.bias_network.state_dict().values()
    ]
    return torch.cat(parts)


def pairwise_cov(vec_i: torch.Tensor, vec_j: torch.Tensor) -> float:
    """Coefficient-of-variation between two flattened weight vectors.

    Defined as ``||vec_i - vec_j||_2 / mean(||vec_i||_2, ||vec_j||_2)``.

    The mean-of-norms denominator (vs e.g. mean of vec_i+vec_j) keeps
    the CoV scale-invariant under uniform multiplicative rescaling: two
    proportional substrates with the same direction return CoV near 0
    iff they are byte-identical, regardless of overall magnitude.

    Returns
    -------
        Float in ``[0, ∞)``. Identical vectors → 0.0; orthogonal
        unit-norm vectors → √2 ≈ 1.414; opposite-sign vectors of equal
        magnitude → 2.0.

    Raises
    ------
        ValueError: if either norm is zero (both substrates degenerate
            to all-zero — undefined CoV).
        ValueError: if the vectors differ in shape (architecture
            mismatch — can't be paired).
    """
    if vec_i.shape != vec_j.shape:
        msg = (
            f"pairwise_cov: vector shapes differ {tuple(vec_i.shape)} vs "
            f"{tuple(vec_j.shape)}; cannot pair architectures."
        )
        raise ValueError(msg)
    norm_i = float(torch.linalg.vector_norm(vec_i).item())
    norm_j = float(torch.linalg.vector_norm(vec_j).item())
    mean_norm = (norm_i + norm_j) / 2.0
    if mean_norm == 0.0:
        msg = (
            "pairwise_cov: both substrate vectors have zero L2 norm. "
            "Undefined CoV — substrate is degenerate (all-zero weights)."
        )
        raise ValueError(msg)
    diff_norm = float(torch.linalg.vector_norm(vec_i - vec_j).item())
    return diff_norm / mean_norm


def compute_pairwise_cov_matrix(
    seed_vectors: dict[int, torch.Tensor],
) -> list[tuple[int, int, float]]:
    """Return all (seed_i, seed_j, cov) triplets for i < j.

    Returns
    -------
        List of ``(seed_i, seed_j, pairwise_cov)`` sorted by
        ``(seed_i, seed_j)``. Empty if fewer than 2 seeds.
    """
    seeds = sorted(seed_vectors.keys())
    triplets: list[tuple[int, int, float]] = []
    for idx_a, seed_a in enumerate(seeds):
        for seed_b in seeds[idx_a + 1 :]:
            cov = pairwise_cov(seed_vectors[seed_a], seed_vectors[seed_b])
            triplets.append((seed_a, seed_b, cov))
    return triplets


def _feature_range(feature_name: str) -> tuple[float, float]:
    """Realistic production range for a sensory ``BrainParams`` feature.

    Used by ``mean_abs_bias_output`` to sample probe inputs that
    match the runtime input distribution. Sampling uniformly from
    ``[-1, 1]`` for every feature is wrong because two of the
    canonical TEI features (``predator_gradient_strength`` and
    ``food_gradient_strength``) are non-negative in production
    (env emits ``gradient_strength * exp(-d / decay)``). Out-of-
    distribution probes can produce a non-zero mean-abs output via
    odd-symmetric activations even when the substrate is
    output-degenerate on the in-distribution range — a false-pass
    on T4.

    Rules (suffix-first, then fallback):
        - ``*_sin`` / ``*_cos`` → ``[-1, 1]`` (trigonometric derived)
        - ``*_strength`` → ``[0, 1]`` (decay-attenuated, non-negative)
        - default → ``[-1, 1]`` (conservatively symmetric)
    """
    if feature_name.endswith(("_sin", "_cos")):
        return -1.0, 1.0
    if feature_name.endswith("_strength"):
        return 0.0, 1.0
    return -1.0, 1.0


def mean_abs_bias_output(
    substrate: TransgenerationalMemory,
    *,
    num_probes: int = 64,
    rng_seed: int = 424242,
) -> float:
    """Mean ``|bias_network(probe)|`` over a deterministic probe set.

    Surfaces the T4 substrate-magnitude tripwire signal. Probes are
    sampled uniformly from the **per-feature** realistic production
    range — see :func:`_feature_range`. Sampling matches the runtime
    input distribution so a substrate that is output-degenerate
    in-distribution is correctly flagged as failing T4, regardless
    of its behaviour on out-of-distribution inputs.

    Returns
    -------
        Mean absolute output value, averaged over all probes AND all
        action dimensions. Near-zero (< 0.1 by default) indicates the
        bias-network output collapses to the identity-on-logits
        function and the substrate carries no signal.

    Raises
    ------
        ValueError: if substrate has no ``bias_network``.
    """
    if substrate.bias_network is None:
        msg = (
            "mean_abs_bias_output: substrate has bias_network=None; "
            "use logit_bias norm directly for the legacy path."
        )
        raise ValueError(msg)
    in_features = len(substrate.input_features)
    if in_features == 0:
        msg = (
            "mean_abs_bias_output: substrate.input_features is empty but "
            "bias_network is set; substrate is malformed."
        )
        raise ValueError(msg)
    gen = torch.Generator(device="cpu").manual_seed(rng_seed)
    # Per-feature scale + offset to map U[0, 1) to [lo, hi].
    lows: list[float] = []
    spans: list[float] = []
    for feature_name in substrate.input_features:
        lo, hi = _feature_range(feature_name)
        lows.append(lo)
        spans.append(hi - lo)
    lo_tensor = torch.tensor(lows)
    span_tensor = torch.tensor(spans)
    probes = torch.rand((num_probes, in_features), generator=gen) * span_tensor + lo_tensor
    with torch.no_grad():
        outputs = substrate.bias_network(probes)
    return float(outputs.abs().mean().item())


def discover_seed_substrates(
    campaign_root: Path,
    *,
    arm: str = "tei_on",
) -> dict[int, Path]:
    """Find each calibration seed's F0 ``.tei.pt`` substrate file.

    Walks ``<campaign_root>/<arm>/seed-<N>/<session>/inheritance/
    gen-000/genome-*.tei.pt`` and returns ``{seed: path}``. Each seed
    is expected to have exactly one F0 elite ``.tei.pt`` (one elite per
    F0 generation). If multiple are found (e.g. resumed session), the
    most recently modified is chosen.

    Returns
    -------
        ``{seed: path}`` keyed by integer seed parsed from
        ``seed-<N>`` directory name. Empty if no substrates discovered.
    """
    out: dict[int, Path] = {}
    arm_dir = campaign_root / arm
    if not arm_dir.is_dir():
        logger.warning("Arm directory %s missing under campaign root.", arm_dir)
        return out
    for seed_dir in sorted(arm_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed-"):
            continue
        try:
            seed = int(seed_dir.name.split("-", 1)[1])
        except (IndexError, ValueError):
            logger.warning("Skipping non-seed directory: %s", seed_dir)
            continue
        candidates = sorted(
            seed_dir.rglob("inheritance/gen-000/genome-*.tei.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            logger.warning(
                "No F0 .tei.pt under %s for seed=%d; calibration smoke may not "
                "have completed for this seed.",
                seed_dir,
                seed,
            )
            continue
        out[seed] = candidates[0]
    return out


def load_seed_vectors(
    substrate_paths: dict[int, Path],
) -> tuple[dict[int, torch.Tensor], dict[int, TransgenerationalMemory]]:
    """Load each seed's substrate and flatten its bias_network state_dict.

    Returns
    -------
        ``(seed_vectors, seed_substrates)`` where ``seed_vectors`` maps
        seed → flattened state-dict tensor and ``seed_substrates`` maps
        seed → the loaded substrate (kept for T4 magnitude probing).
    """
    seed_vectors: dict[int, torch.Tensor] = {}
    seed_substrates: dict[int, TransgenerationalMemory] = {}
    for seed, path in substrate_paths.items():
        substrate = load_substrate(path)
        seed_substrates[seed] = substrate
        seed_vectors[seed] = _flatten_state_dict(substrate)
    return seed_vectors, seed_substrates


def evaluate_diversity(
    seed_vectors: dict[int, torch.Tensor],
    seed_substrates: dict[int, TransgenerationalMemory],
    *,
    diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
    magnitude_threshold: float = DEFAULT_MAGNITUDE_THRESHOLD,
) -> dict:
    """Run T2 + T4 checks and return a structured verdict dict.

    Returns
    -------
        ``{
            "n_seeds": int,
            "pairwise_covs": [(seed_a, seed_b, cov), ...],
            "min_pairwise_cov": float,
            "mean_pairwise_cov": float,
            "diversity_threshold": float,
            "diversity_pass": bool,
            "per_seed_magnitudes": {seed: float, ...},
            "min_magnitude": float,
            "magnitude_threshold": float,
            "magnitude_pass": bool,
            "overall_pass": bool,
        }``.

    ``overall_pass`` requires BOTH T2 (diversity) AND T4 (magnitude)
    to hold. A single ``False`` from either trips the script's exit
    code 1.
    """
    triplets = compute_pairwise_cov_matrix(seed_vectors)
    n_seeds = len(seed_vectors)
    if n_seeds < 2 or not triplets:
        # Single-seed or empty input — diversity is undefined. We
        # conservatively fail closed (overall_pass=False) and surface
        # the n_seeds=0/1 condition so the operator sees the
        # underlying issue (no calibration seeds completed) rather
        # than a misleading "pass".
        min_cov = math.nan
        mean_cov = math.nan
        diversity_pass = False
    else:
        covs = [c for _, _, c in triplets]
        min_cov = min(covs)
        mean_cov = sum(covs) / len(covs)
        diversity_pass = min_cov >= diversity_threshold
    per_seed_mag = {seed: mean_abs_bias_output(sub) for seed, sub in seed_substrates.items()}
    min_mag = min(per_seed_mag.values()) if per_seed_mag else math.nan
    magnitude_pass = (not math.isnan(min_mag)) and min_mag >= magnitude_threshold
    overall_pass = diversity_pass and magnitude_pass
    return {
        "n_seeds": n_seeds,
        "pairwise_covs": triplets,
        "min_pairwise_cov": min_cov,
        "mean_pairwise_cov": mean_cov,
        "diversity_threshold": diversity_threshold,
        "diversity_pass": diversity_pass,
        "per_seed_magnitudes": per_seed_mag,
        "min_magnitude": min_mag,
        "magnitude_threshold": magnitude_threshold,
        "magnitude_pass": magnitude_pass,
        "overall_pass": overall_pass,
    }


def _format_summary_text(verdict: dict) -> str:
    """Render the verdict as human-readable text for stderr."""
    lines = [
        "Substrate diversity tripwire (T2) + magnitude tripwire (T4):",
        f"  Seeds evaluated:        {verdict['n_seeds']}",
        f"  Min pairwise CoV:       {verdict['min_pairwise_cov']:.4f} "
        f"(threshold >= {verdict['diversity_threshold']:.4f})",
        f"  Mean pairwise CoV:      {verdict['mean_pairwise_cov']:.4f}",
        f"  T2 diversity pass:      {verdict['diversity_pass']}",
        f"  Min |bias_net output|:  {verdict['min_magnitude']:.4f} "
        f"(threshold >= {verdict['magnitude_threshold']:.4f})",
        f"  T4 magnitude pass:      {verdict['magnitude_pass']}",
        f"  Overall (T2 AND T4):    {verdict['overall_pass']}",
    ]
    return "\n".join(lines)


def _write_csv(verdict: dict, csv_path: Path) -> None:
    """Persist the pairwise-CoV table and per-seed magnitudes to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("section,key_a,key_b,value\n")
        for seed_a, seed_b, cov in verdict["pairwise_covs"]:
            handle.write(f"pairwise_cov,seed-{seed_a},seed-{seed_b},{cov:.6f}\n")
        for seed, mag in sorted(verdict["per_seed_magnitudes"].items()):
            handle.write(f"mean_abs_bias_output,seed-{seed},,{mag:.6f}\n")
        handle.write(
            f"summary,min_pairwise_cov,,{verdict['min_pairwise_cov']:.6f}\n",
        )
        handle.write(
            f"summary,mean_pairwise_cov,,{verdict['mean_pairwise_cov']:.6f}\n",
        )
        handle.write(
            f"summary,min_magnitude,,{verdict['min_magnitude']:.6f}\n",
        )
        handle.write(
            f"summary,diversity_pass,,{int(verdict['diversity_pass'])}\n",
        )
        handle.write(
            f"summary,magnitude_pass,,{int(verdict['magnitude_pass'])}\n",
        )
        handle.write(
            f"summary,overall_pass,,{int(verdict['overall_pass'])}\n",
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Substrate-diversity tripwire (T2) for the M6.9+ TEI re-evaluation "
            "calibration smoke. Computes pairwise CoV across calibration seeds' "
            "bias_network state_dicts + mean |bias output| (T4 surrogate)."
        ),
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help=(
            "Campaign output root, e.g. evolution_results/m69_transgenerational/. "
            "Substrates are discovered at <root>/<arm>/seed-<N>/<session>/"
            "inheritance/gen-000/genome-*.tei.pt."
        ),
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="tei_on",
        help=(
            "Arm to evaluate (default: tei_on). Only the tei_on arm has a "
            "bias-network substrate; weights_only/control are skipped."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV output path for the pairwise-CoV table + per-seed "
            "magnitudes. Parent directories are created if missing."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DIVERSITY_THRESHOLD,
        help=(
            f"Pairwise-CoV diversity threshold (default {DEFAULT_DIVERSITY_THRESHOLD}). "
            "Below this, the tripwire fails and exit code is 1."
        ),
    )
    parser.add_argument(
        "--magnitude-threshold",
        type=float,
        default=DEFAULT_MAGNITUDE_THRESHOLD,
        help=(
            f"Substrate-magnitude threshold (default {DEFAULT_MAGNITUDE_THRESHOLD}). "
            "Below this, T4 fails and exit code is 1."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns exit code (0 pass, 1 fail)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    substrate_paths = discover_seed_substrates(args.campaign_root, arm=args.arm)
    if not substrate_paths:
        msg = (
            f"No F0 substrates discovered under "
            f"{args.campaign_root}/{args.arm}/seed-*/. Has the calibration smoke "
            f"completed?"
        )
        print(msg, file=sys.stderr)
        print(json.dumps({"overall_pass": False, "reason": "no_substrates_found"}))
        return 1
    seed_vectors, seed_substrates = load_seed_vectors(substrate_paths)
    verdict = evaluate_diversity(
        seed_vectors,
        seed_substrates,
        diversity_threshold=args.threshold,
        magnitude_threshold=args.magnitude_threshold,
    )
    print(_format_summary_text(verdict), file=sys.stderr)
    if args.output_csv is not None:
        _write_csv(verdict, args.output_csv)
        print(f"Wrote pairwise-CoV table to {args.output_csv}", file=sys.stderr)
    # JSON to stdout for machine consumption (campaign shell parses
    # ``overall_pass`` to decide whether to abort).
    json_payload = {
        "n_seeds": verdict["n_seeds"],
        "min_pairwise_cov": verdict["min_pairwise_cov"],
        "mean_pairwise_cov": verdict["mean_pairwise_cov"],
        "diversity_threshold": verdict["diversity_threshold"],
        "diversity_pass": verdict["diversity_pass"],
        "min_magnitude": verdict["min_magnitude"],
        "magnitude_threshold": verdict["magnitude_threshold"],
        "magnitude_pass": verdict["magnitude_pass"],
        "overall_pass": verdict["overall_pass"],
    }
    print(json.dumps(json_payload))
    return 0 if verdict["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
