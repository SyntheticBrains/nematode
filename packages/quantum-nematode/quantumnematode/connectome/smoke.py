"""Smoke-test forward pass — confirms the data model supports learned-weight compute.

Runs a single random-weight forward pass through the connectome topology
and verifies the output is finite, non-degenerate, and consumes both
connection types correctly (chemical strict-mask + fixed gap-junction
weights).

This is a data-model validation, not a brain implementation. It does NOT
register a brain type, hook into the agent loop, or touch the env. The
actual brain wiring is downstream work.

Run as a CLI smoke check:

    uv run python -m quantumnematode.connectome.smoke
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite

if TYPE_CHECKING:
    from quantumnematode.connectome.model import Connectome


_DEGENERATE_VARIANCE_THRESHOLD: float = 1e-6
"""Below this variance the forward pass is considered degenerate / saturated."""


def _build_adjacency(
    connectome: Connectome,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Build the combined weight matrix (chemical + gap-junction) over the connectome.

    Chemical synapses are strict-masked random weights: only edges that
    exist in the connectome get a non-zero entry, sampled from
    ``N(0, 1/sqrt(fan_in))`` where ``fan_in`` is the in-degree of the
    postsynaptic neuron. This prevents tanh saturation given that some
    *C. elegans* neurons have ~50 chemical inputs.

    Gap junctions participate at their fixed connectome-reported weights
    (junction counts), normalised by the same fan-in factor so the two
    connection types contribute on a comparable scale to the
    pre-activation sum. The gap-junction matrix is symmetric.

    Returns the combined adjacency matrix ``W`` and the neuron-name list
    ordered by row/column index.
    """
    names = sorted(connectome.neurons)
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}

    # Strict-mask: count fan-in per post neuron
    chem_in_degree = np.zeros(n, dtype=np.int64)
    for syn in connectome.chemical_synapses:
        chem_in_degree[idx[syn.post]] += 1

    # 1 / sqrt(fan_in) per post column; zero-fan-in cols stay at zero.
    # Guard the division so np.where's "false" branch doesn't compute a
    # bogus 1/0 (which would emit a RuntimeWarning).
    safe_chem_in = np.where(chem_in_degree > 0, chem_in_degree, 1)
    chem_scale = np.where(chem_in_degree > 0, 1.0 / np.sqrt(safe_chem_in), 0.0)

    w_chem = np.zeros((n, n), dtype=np.float64)
    for syn in connectome.chemical_synapses:
        i = idx[syn.pre]
        j = idx[syn.post]
        w_chem[i, j] = rng.normal(loc=0.0, scale=chem_scale[j])

    # Gap junctions: symmetric matrix, fixed weights from connectome counts
    w_gap = np.zeros((n, n), dtype=np.float64)
    for gj in connectome.gap_junctions:
        i = idx[gj.neuron_a]
        j = idx[gj.neuron_b]
        w_gap[i, j] = float(gj.weight)
        w_gap[j, i] = float(gj.weight)

    # Apply the same fan-in normalisation to gap junctions so the two
    # contributions are on a comparable scale.
    gap_in_degree = (w_gap != 0).sum(axis=0)
    safe_gap_in = np.where(gap_in_degree > 0, gap_in_degree, 1)
    gap_scale = np.where(gap_in_degree > 0, 1.0 / np.sqrt(safe_gap_in), 0.0)
    w_gap = w_gap * gap_scale[np.newaxis, :]

    return w_chem + w_gap, names


def run_forward_pass(connectome: Connectome, *, seed: int = 0) -> np.ndarray:
    """Run one PPO-shaped forward pass on the connectome topology.

    The pass computes ``output = tanh((W_chem + W_gap).T @ x)`` where
    ``x`` is a synthetic seeded random input vector and ``W_chem`` /
    ``W_gap`` are the chemical-synapse / gap-junction matrices described
    in ``_build_adjacency``. The transpose is biologically correct:
    ``W[i, j]`` stores the weight from pre neuron ``i`` to post neuron
    ``j``, so each post neuron's pre-activation sum is
    ``sum_i(W[i, j] * x[i])`` = ``(W.T @ x)[j]``. The returned array
    contains the values at motor-neuron rows only.

    Raises
    ------
    ValueError
        If the connectome has zero chemical synapses (sanity guard
        against silent load failure that would otherwise return a
        degenerate output).
    """
    if not connectome.chemical_synapses:
        msg = (
            "Connectome has zero chemical synapses; cannot run a meaningful "
            "forward pass. This likely indicates a silent load failure or an "
            "empty test fixture."
        )
        raise ValueError(msg)

    rng = np.random.default_rng(seed)

    w_combined, names = _build_adjacency(connectome, rng=rng)
    n = len(names)
    x = rng.normal(loc=0.0, scale=1.0, size=n)

    preact = w_combined.T @ x  # shape (n,) — each post neuron's pre-activation sum
    activation = np.tanh(preact)

    motor_indices = [
        i for i, name in enumerate(names) if connectome.neurons[name].cell_class == "motor"
    ]
    return activation[motor_indices]


def main() -> int:
    """CLI entrypoint. Run the smoke pass on the Cook 2019 hermaphrodite connectome."""
    print("Loading Cook 2019 hermaphrodite connectome...")  # noqa: T201
    connectome = load_cook_2019_hermaphrodite()
    print(  # noqa: T201
        f"  Loaded: {len(connectome.neurons)} neurons, "
        f"{len(connectome.chemical_synapses)} chemical synapses, "
        f"{len(connectome.gap_junctions)} gap junctions.",
    )

    print("Running forward pass (seed=0)...")  # noqa: T201
    output = run_forward_pass(connectome, seed=0)

    print(f"  Output shape: {output.shape}")  # noqa: T201
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")  # noqa: T201
    print(f"  Output variance across motor rows: {output.var():.6f}")  # noqa: T201

    if not np.isfinite(output).all():
        print("  FAIL: output contains non-finite values.")  # noqa: T201
        return 1
    if output.var() < _DEGENERATE_VARIANCE_THRESHOLD:
        print(  # noqa: T201
            f"  FAIL: output variance below {_DEGENERATE_VARIANCE_THRESHOLD} "
            "(degenerate / saturated).",
        )
        return 1

    print("  PASS: output is finite and non-degenerate.")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
