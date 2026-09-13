"""Build checkpoints that differ from an arm's own initialisation in the motor readout alone.

R.1c closed the perturbation dimension and eliminated two of the three tensors PPO trains that the
rule cannot write. The readout is the one left, and this prepares the arms that test it.

**Only the readout moves.** Each checkpoint is the baseline arm's own state at that seed with one
tensor substituted. A wholesale load of a PPO checkpoint would also bring:

* ``log_std`` -- changing the action noise the rule runs at, an axis R.1c swept and pinned;
* ``w_chem`` -- making the arm a clone assay, which is a different experiment;
* ``food_gains`` -- an axis R.1c eliminated, and a second simultaneous change.

The constraint is load-bearing for a second reason: the rule re-anchors its homeostatic norm targets
from the weights it is loaded with, so a checkpoint carrying a different ``w_chem`` would silently
move those targets too.

Three sources:

* ``anatomical`` -- the arm's own readout, unchanged. Not an experimental arm: it exists so the
  load path itself can be shown inert, which is what licenses reusing R.1c's committed pair as the
  comparator instead of re-running it.
* ``ppo`` -- harvested from that seed's PPO run on the same cell, at the same action scale.
* ``rotated`` -- a random direction at the same Frobenius norm as that seed's ``ppo`` readout. This is
  the arm that makes a positive result interpretable: without it, "PPO found a good readout" cannot be
  told from "the anatomical prior is bad and almost anything beats it".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.weights import save_weights
from quantumnematode.utils.config_loader import load_simulation_config

if TYPE_CHECKING:
    from collections.abc import Sequence

_ANALYSIS = Path(__file__).resolve().parent / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

SOURCES = ("anatomical", "ppo", "rotated")
READOUT_KEY = "readout"
# Everything else must come through untouched; asserted per file before it is written.
PROTECTED = ("w_chem", "food_gains", "log_std", "m_chem", "g_gap", "chem_sign")


def _harvest_readout(harvest_dir: Path, seed: int) -> torch.Tensor:
    """Read the readout from that seed's PPO harvest, resolved through its experiment record."""
    from l4_panel import (  # pyright: ignore[reportMissingImports]
        EXPERIMENTS,
        _experiment_json,
    )

    logs = sorted((harvest_dir / "logs").glob(f"*-seed{seed}.log"))
    if not logs:
        msg = f"no harvest log for seed {seed} under {harvest_dir}"
        raise FileNotFoundError(msg)
    if len(logs) > 1:
        msg = f"{len(logs)} harvest logs for seed {seed} under {harvest_dir}: {[p.name for p in logs]}"
        raise ValueError(msg)
    experiment = _experiment_json(logs[0].read_text(), EXPERIMENTS)
    exports = experiment.get("exports_path") if experiment else None
    if not exports:
        msg = (
            f"harvest run for seed {seed} has no exports_path; was it run with --track-experiment?"
        )
        raise FileNotFoundError(msg)
    final = Path(exports) / "weights" / "final.pt"
    if not final.is_file():
        msg = f"harvest run for seed {seed} has no weights at {final}"
        raise FileNotFoundError(msg)
    topology = torch.load(final, weights_only=True).get("topology")
    if not isinstance(topology, dict) or READOUT_KEY not in topology:
        msg = f"harvest checkpoint for seed {seed} carries no {READOUT_KEY!r}"
        raise ValueError(msg)
    return topology[READOUT_KEY].detach().clone()


def _rotate(reference: torch.Tensor, seed: int) -> torch.Tensor:
    """Draw a random direction at the reference's Frobenius norm, from a seeded generator.

    Same norm rather than same distribution: a rotation isolates the readout's DIRECTION while
    leaving its scale -- which interacts with the action distribution -- where the reference put it.
    """
    generator = torch.Generator().manual_seed(seed)
    draw = torch.randn(reference.shape, generator=generator, dtype=reference.dtype)
    norm = float(draw.norm())
    if norm == 0.0:  # pragma: no cover - probability zero
        msg = "drew a zero matrix"
        raise ValueError(msg)
    return draw * (float(reference.norm()) / norm)


def _replacement_readout(
    source: str,
    original: torch.Tensor,
    seed: int,
    harvest_dir: Path | None,
) -> torch.Tensor:
    """Select the readout this source substitutes."""
    if source == "anatomical":
        return original.clone()
    if source in ("ppo", "rotated"):
        if harvest_dir is None:
            msg = f"source {source!r} needs --harvest-dir"
            raise ValueError(msg)
        harvested = _harvest_readout(harvest_dir, seed)
        return harvested if source == "ppo" else _rotate(harvested, seed)
    msg = f"unknown source {source!r}"  # pragma: no cover - argparse restricts the choices
    raise ValueError(msg)


def prepare(
    config_path: Path,
    source: str,
    seed: int,
    out_dir: Path,
    harvest_dir: Path | None,
) -> Path:
    """Write one checkpoint, and its provenance sidecar, for one source and one seed."""
    simulation = load_simulation_config(str(config_path))
    if simulation.brain is None:
        msg = f"{config_path} declares no brain"
        raise ValueError(msg)
    brain_config = simulation.brain.config
    if not isinstance(brain_config, ConnectomePPOBrainConfig):
        # The readout this substitutes is a connectome tensor; any other brain has no such thing.
        msg = f"{config_path} is not a connectome arm (brain config {type(brain_config).__name__})"
        raise TypeError(msg)
    brain_config.seed = seed
    brain = ConnectomePPOBrain(config=brain_config, device=DeviceType.CPU)

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"readout_{source}_seed{seed}.pt"
    save_weights(brain, target)
    checkpoint = torch.load(target, weights_only=True)
    topology = checkpoint["topology"]
    original = topology[READOUT_KEY].detach().clone()

    replacement = _replacement_readout(source, original, seed, harvest_dir)

    if replacement.shape != original.shape:
        msg = f"readout shape {tuple(replacement.shape)} != the arm's {tuple(original.shape)}"
        raise ValueError(msg)

    before = {k: topology[k].detach().clone() for k in PROTECTED if k in topology}
    topology[READOUT_KEY] = replacement.to(original.dtype)
    # The whole point of the file: one tensor differs and nothing else does.
    for key, saved in before.items():
        if not bool(torch.equal(topology[key], saved)):
            msg = f"preparation changed {key!r}, which must come through untouched"
            raise AssertionError(msg)
    torch.save(checkpoint, target)

    cosine = float(
        torch.nn.functional.cosine_similarity(
            replacement.reshape(-1).float(),
            original.reshape(-1).float(),
            dim=0,
        ),
    )
    (out_dir / f"readout_{source}_seed{seed}.json").write_text(
        json.dumps(
            {
                "source": source,
                "seed": seed,
                "config": str(config_path),
                "harvest_dir": str(harvest_dir) if harvest_dir else None,
                "readout_norm_before": float(original.norm()),
                "readout_norm_after": float(replacement.norm()),
                "cosine_to_anatomical": cosine,
                "protected_tensors": list(before),
            },
            indent=2,
        )
        + "\n",
    )
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare every seed's checkpoint for one source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="the baseline arm's config")
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--seeds", default="1-8")
    parser.add_argument("--harvest-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    low, _, high = args.seeds.partition("-")
    for seed in range(int(low), int(high or low) + 1):
        written = prepare(args.config, args.source, seed, args.out_dir, args.harvest_dir)
        print(f"  {args.source:<11} seed {seed}: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
