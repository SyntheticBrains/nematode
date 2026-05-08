"""Curate the prey reference bundle for the co-evolution loop.

This is a one-shot data-preparation utility: it loads the per-seed final
elite LSTMPPO weight checkpoints from a prior single-population
campaign (typically the lamarckian-LSTMPPO-klinotaxis-predator pilot at
``artifacts/logbooks/013/m3_lamarckian_pilot/lamarckian/seed-{42,43,44,45}/``),
flattens each elite's weights via the same ``MLPPPOEncoder`` /
``LSTMPPOEncoder`` flatten path the genome encoder uses, and emits one
JSON bundle:

* ``configs/evolution/coevolution_warmstart_prey/seed_{42..45}.json``:
  serves BOTH the prey gen-0 warm-start anchor
  (``CMAESOptimizer(x0=...)``, consumed by ``_load_prey_warmstart``)
  AND the generality probe's prey-side held-out set (consumed by
  ``_load_held_out_prey_bundle``). The two roles share the same
  source genomes; one bundle on disk avoids ~5 MB of byte-identical
  duplication.

Output schema (matches ``CoevolutionLoop._load_prey_warmstart`` /
``_load_held_out_prey_bundle`` expectations):

```
{
  "genome_id": str,           # source genome id from inheritance/*.pt filename
  "generation": int,          # source generation (final M3 generation = 19)
  "fitness": float,           # source elite fitness from lineage.csv
  "params": list[float],      # flattened weight vector matching encoder.genome_dim
  "brain_config": {           # provenance for downstream introspection
    "name": "lstmppo",
    "comment": "..."
  }
}
```

Usage::

    uv run python scripts/campaigns/curate_coevolution_prey_bundles.py \
        --source-config configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml \
        --source-root artifacts/logbooks/013/m3_lamarckian_pilot/lamarckian \
        --out-root configs/evolution/

The ``--source-config`` argument must be the YAML used to RUN the source
campaign (the same one the .pt checkpoints were saved against): the
script needs the brain shape (``actor_hidden_dim``, ``critic_hidden_dim``,
``num_hidden_layers``, ``rnn_type``, ``lstm_hidden_dim``, sensory modules)
to construct a fresh brain whose component shapes match the saved weights.

Re-running the script is safe: it overwrites existing JSONs in the
output dir and rewrites the README. No state is carried between runs.

Note on bundle size: the M3 lamarckian campaign saved one elite ``.pt``
file per seed (intermediates were GC'd), so the bundle ships
4 distinct genomes (one per source seed). Coevolution YAMLs that target
this bundle SHOULD set ``held_out_size: 4`` (default 8 in
``CoevolutionConfig`` is generous; the loader gracefully samples
WITH replacement when ``held_out_size > len(bundle)``, so an oversized
held_out_size still works but with sample repetition).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from quantumnematode.brain.weights import load_weights
from quantumnematode.evolution.brain_factory import (
    instantiate_brain_from_sim_config,
)
from quantumnematode.evolution.encoders import (
    _as_weight_persistence,
    _flatten_components,
    _select_genome_components,
)
from quantumnematode.utils.config_loader import load_simulation_config

logger = logging.getLogger(__name__)


def _find_elite_pt(seed_dir: Path) -> Path:
    """Return the single ``inheritance/*.pt`` checkpoint under a seed dir."""
    inheritance_dir = seed_dir / "inheritance"
    if not inheritance_dir.is_dir():
        msg = (
            f"Inheritance dir not found at {inheritance_dir!r}. The source "
            "campaign must have written per-genome elite weight checkpoints "
            "under a per-seed inheritance/ subdir."
        )
        raise FileNotFoundError(msg)
    pt_files = sorted(inheritance_dir.glob("genome-*.pt"))
    if not pt_files:
        msg = (
            f"No genome-*.pt files found under {inheritance_dir!r}. The "
            "source campaign should have saved at least the final elite."
        )
        raise FileNotFoundError(msg)
    if len(pt_files) > 1:
        # Pick the most recent by mtime — assumes the saver only updates
        # the elite file as the campaign progresses, so the latest write
        # is the best genome. Multi-elite campaigns should specialise this.
        pt_files.sort(key=lambda p: p.stat().st_mtime)
        logger.warning(
            "Multiple .pt files in %s; selecting the most recently modified: %s",
            inheritance_dir,
            pt_files[-1].name,
        )
    return pt_files[-1]


def _lookup_genome_metadata_in_lineage(
    lineage_csv: Path,
    genome_id: str,
) -> tuple[int, float] | None:
    """Look up ``(generation, fitness)`` for a genome_id in lineage.csv.

    Returns None when the genome is not found (pre-fix lineage CSVs may
    not record GC'd genome rows). Caller falls back to placeholder
    metadata in that case — the params vector is still authentic.
    """
    if not lineage_csv.exists():
        return None
    with lineage_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("child_id") == genome_id:
                return int(row["generation"]), float(row["fitness"])
    return None


def curate_seed(
    *,
    sim_config_path: Path,
    seed_dir: Path,
    seed: int,
    bundle_out: Path,
) -> None:
    """Curate one seed: extract weights, write the unified bundle JSON.

    Parameters
    ----------
    sim_config_path
        Source-campaign YAML; used to build a fresh brain whose
        component shapes match the saved checkpoint.
    seed_dir
        Source per-seed dir (must contain ``inheritance/genome-*.pt``
        and ideally ``lineage.csv``).
    seed
        Source campaign seed (e.g. 42); embedded in the output
        ``brain_config.comment`` for provenance.
    bundle_out
        Output JSON path. Single file per seed serves both the
        warm-start anchor and the held-out probe roles (same source
        genome, two roles).
    """
    pt_path = _find_elite_pt(seed_dir)
    genome_id = pt_path.stem.removeprefix("genome-")

    sim_config = load_simulation_config(str(sim_config_path))
    # Build a brain whose component shapes match the saved checkpoint.
    # `seed=None` is fine — the orthogonal-init values are about to be
    # overwritten by `load_weights`.
    brain = instantiate_brain_from_sim_config(sim_config, seed=None)
    load_weights(brain, pt_path)

    # Flatten components via the same path encoders use, so the resulting
    # `params` list is byte-identical to what `MLPPPOEncoder.initial_genome`
    # would produce on a brain with these weights. NON_GENOME_COMPONENTS
    # (optimizer state + training_state) are filtered out.
    wp = _as_weight_persistence(brain)
    components = _select_genome_components(wp.get_weight_components())
    params, _ = _flatten_components(components)
    flat_params = [float(v) for v in params]

    metadata = _lookup_genome_metadata_in_lineage(
        seed_dir / "lineage.csv",
        genome_id,
    )
    if metadata is None:
        generation, fitness = -1, -1.0
        logger.warning(
            "Genome %s not found in %s/lineage.csv; recording "
            "generation=-1, fitness=-1.0 as a placeholder.",
            genome_id,
            seed_dir,
        )
    else:
        generation, fitness = metadata

    payload = {
        "genome_id": genome_id,
        "generation": generation,
        "fitness": fitness,
        "params": flat_params,
        "brain_config": {
            "name": sim_config.brain.name if sim_config.brain else "lstmppo",
            "comment": (
                f"Source: {seed_dir.relative_to(seed_dir.parent.parent.parent.parent)}/inheritance/{pt_path.name}; "
                f"source-campaign seed={seed}; "
                f"params length {len(flat_params)} matches encoder genome_dim "
                "for the brain shape declared in the source-campaign YAML."
            ),
        },
    }

    bundle_out.parent.mkdir(parents=True, exist_ok=True)
    bundle_out.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Seed %s: wrote %d-float bundle JSON -> %s (genome_id=%s, gen=%s, fitness=%.4f)",
        seed,
        len(flat_params),
        bundle_out,
        genome_id,
        generation,
        fitness,
    )


def _write_readme(
    *,
    out_dir: Path,
    seeds: list[int],
    source_config: Path,
    source_root: Path,
) -> None:
    """Emit a small README documenting the bundle's provenance + dual role."""
    out_dir.mkdir(parents=True, exist_ok=True)
    readme_path = out_dir / "README.md"
    readme_path.write_text(
        f"""# Coevolution prey reference bundle

Each `seed_{{N}}.json` is one prey LSTMPPO elite genome with TWO roles
in the co-evolution loop, both consumed from this single bundle:

1. **Gen-0 warm-start anchor** for the prey CMA-ES (per-seed
   `coevolution.prey_gen0_seed_path` in the YAML, consumed by
   `CoevolutionLoop._load_prey_warmstart`). Each campaign seed loads
   its own anchor (template is `seed_<run_seed>.json`).
2. **Held-out probe opponents** for the co-evolution generality
   probe. `CoevolutionLoop._load_held_out_prey_bundle` walks the same
   directory and loads ALL JSONs (one per seed) into the held-out
   set, then samples `held_out_size` of them per probe firing.

The two roles share the same source genomes by design — one bundle
on disk avoids ~5 MB of byte-identical duplication between separate
warmstart/ and held-out/ directories.

Files committed to repo (vs `artifacts/`) so a fresh checkout can run
the campaign reproducibly. Routed through Git LFS via the repo's
`configs/**/*.json` rule in `.gitattributes`.

## Provenance

Source campaign: `{source_config.name}` driven by the matching campaign
script. Per-seed elite LSTMPPO weight checkpoints under
`{source_root}/seed-{{42..45}}/inheritance/genome-*.pt` (one elite saved
per seed; intermediate checkpoints GC'd by the source campaign).

Seeds covered: {seeds}.

## Format

Each JSON file matches the `CoevolutionLoop._load_prey_warmstart`
schema:

```json
{{
  "genome_id": "<uuid>",
  "generation": <int>,
  "fitness": <float>,
  "params": [<float>, ...],
  "brain_config": {{
    "name": "lstmppo",
    "comment": "..."
  }}
}}
```

The `params` array is the flattened weight vector matching the
encoder's `genome_dim` for the source-campaign brain shape (LSTMPPO +
klinotaxis sensing as configured in the source YAML). Co-evolution YAMLs
that target this bundle MUST use the same brain shape (or a compatible
encoder that produces the same `genome_dim`).

## Regenerate

```sh
uv run python scripts/campaigns/curate_coevolution_prey_bundles.py \\
    --source-config {source_config} \\
    --source-root {source_root} \\
    --out-root configs/evolution/
```
""",
    )
    logger.info("Wrote README -> %s", readme_path)


def main() -> int:
    """Curate both bundles for all source seeds."""
    parser = argparse.ArgumentParser(
        description="Curate prey held-out + warmstart bundles for co-evolution.",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        required=True,
        help=(
            "Source-campaign YAML (the one .pt checkpoints were saved against). "
            "Must declare the same brain shape as the saved weights."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Per-seed dirs root (expects seed-42/, seed-43/, ... underneath).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("configs/evolution"),
        help=(
            "Output root containing the unified bundle subdir "
            "(coevolution_warmstart_prey/). Both warmstart anchors and "
            "held-out probe opponents read from this same dir."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45],
        help="Source-campaign seeds to curate (one elite per seed).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.source_config.is_file():
        logger.error("--source-config not found: %s", args.source_config)
        return 1
    if not args.source_root.is_dir():
        logger.error("--source-root is not a directory: %s", args.source_root)
        return 1

    bundle_dir = args.out_root / "coevolution_warmstart_prey"

    completed_seeds: list[int] = []
    for seed in args.seeds:
        seed_dir = args.source_root / f"seed-{seed}"
        if not seed_dir.is_dir():
            logger.error("Seed dir missing, aborting: %s", seed_dir)
            return 1
        bundle_out = bundle_dir / f"seed_{seed}.json"
        # Pre-delete any existing per-seed file so a stale JSON from a
        # prior run can't get mixed in if curate_seed fails mid-write
        # (we abort below on any exception, but defensive removal here
        # also avoids partial overwrites if `write_text` is interrupted).
        if bundle_out.exists():
            bundle_out.unlink()
        try:
            curate_seed(
                sim_config_path=args.source_config,
                seed_dir=seed_dir,
                seed=seed,
                bundle_out=bundle_out,
            )
        except (FileNotFoundError, ValueError, RuntimeError):
            # Abort instead of continuing: a partial bundle (some
            # seeds curated, others stale or missing) would silently
            # mislead downstream loaders. Fail loud and let the user
            # re-run after fixing the source.
            logger.exception("Seed %s failed; aborting curation", seed)
            return 1
        completed_seeds.append(seed)

    _write_readme(
        out_dir=bundle_dir,
        seeds=completed_seeds,
        source_config=args.source_config,
        source_root=args.source_root,
    )
    logger.info(
        "Curated %d seeds; bundle at %s (warmstart + held-out roles)",
        len(completed_seeds),
        bundle_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
