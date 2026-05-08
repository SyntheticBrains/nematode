# pragma: no cover
r"""Run a co-evolution campaign via :class:`CoevolutionLoop`.

The capability spec lives at ``openspec/specs/co-evolution/`` (added by
the ``add-coevolution-arms-race`` change). Single-population analogue
is :mod:`scripts.run_evolution`.

Examples
--------
::

    # Smoke pilot: ~60 sec end-to-end validation.
    uv run python scripts/run_coevolution.py \
        --config configs/evolution/coevolution_smoke.yml \
        --seed 42

    # Pilot arm A (heuristic-imitation pretrain, seed=42).
    uv run python scripts/run_coevolution.py \
        --config configs/evolution/coevolution_pilot_arm_a.yml \
        --seed 42 \
        --output-dir evolution_results/m5_coevolution_pilot/arm_a

    # Resume an interrupted run.
    uv run python scripts/run_coevolution.py \
        --config configs/evolution/coevolution_pilot_arm_a.yml \
        --seed 42 \
        --resume evolution_results/m5_coevolution_pilot/arm_a/<session>

Output layout (per run, mirrors the CoevolutionLoop checkpoint format):

    <output_dir>/<session_id>/
        prey/
            checkpoint.pkl
            lineage.csv
        predator/
            checkpoint.pkl
            lineage.csv
        coevolution_state.json
        coevolution_rng.pkl
        champion_history.json
        generality_probe.csv

The ``--resume`` flag points at the session directory (NOT a checkpoint
file) since the loop reads five files. The driver passes ``resume=True``
to :meth:`CoevolutionLoop.run`; the loop validates cross-file
consistency and refuses to resume on partial saves.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from quantumnematode.evolution.coevolution import CoevolutionLoop
from quantumnematode.logging_config import configure_file_logging, logger
from quantumnematode.utils.config_loader import load_simulation_config
from quantumnematode.utils.session import generate_session_id


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Drive CoevolutionLoop from a YAML config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (must include a `coevolution:` block).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Master seed for the loop. Required for reproducibility "
            "(holds-out RNG, optimiser-seed derivation, eval seeds). "
            "When None, falls back to the YAML's top-level `seed:` field "
            "or 0."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evolution_results",
        help=(
            "Per-campaign root directory. The driver appends a fresh "
            "session id (or, on --resume, reuses the resumed session id)."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to an existing session directory to resume from "
            "(the dir containing prey/, predator/, coevolution_state.json, "
            "coevolution_rng.pkl, champion_history.json). "
            "Mutually exclusive with starting fresh; reuses the "
            "session id from the dir name."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default INFO).",
    )
    return parser.parse_args()


def _resolve_session_dir(
    output_dir_arg: str,
    resume_arg: str | None,
) -> tuple[str, Path] | None:
    """Pick the session id + output directory.

    On resume, derive both the session id and the output dir from the
    resumed dir's path so checkpoint cross-file consistency is preserved
    (lineage CSVs append to the existing files; champion_history rolls
    forward with the existing schema). When --output-dir is also passed
    on resume, warn but honour the resumed dir's location.
    """
    if resume_arg is None:
        session_id = generate_session_id()
        return session_id, Path(output_dir_arg) / session_id

    resume_path = Path(resume_arg).resolve()
    if not resume_path.is_dir():
        logger.error("--resume path is not a directory: %s", resume_path)
        return None
    # Full checkpoint-set sanity: refuse to resume if any of the four
    # required files are missing. Firing here avoids paying the
    # `CoevolutionLoop.__init__` cost (incl. the ~30s heuristic-imitation
    # pretrain on arm A) before `_load_checkpoint` discovers the
    # incomplete set. The rng-pickle is the canonical "checkpoint
    # complete" signal (written LAST by `_save_checkpoint`); if any of
    # the other three are missing while the rng-pickle is present,
    # that's a torn save (concurrent writer, manual deletion, partial
    # restore from backup) — caught here either way.
    required_files = [
        resume_path / "prey" / "checkpoint.pkl",
        resume_path / "predator" / "checkpoint.pkl",
        resume_path / "coevolution_state.json",
        resume_path / "coevolution_rng.pkl",
    ]
    missing = [str(p) for p in required_files if not p.is_file()]
    if missing:
        logger.error(
            "Cannot resume from %s — required checkpoint files missing:\n  %s",
            resume_path,
            "\n  ".join(missing),
        )
        return None
    explicit_root = Path(output_dir_arg).resolve()
    if explicit_root != resume_path.parent:
        logger.warning(
            "--output-dir %s ignored on resume; writing into the "
            "resumed session directory %s instead.",
            output_dir_arg,
            resume_path,
        )
    return resume_path.name, resume_path


def main() -> int:  # noqa: PLR0911 — sequential CLI entrypoint with distinct early-exit error paths (config load, missing coevolution block, resume-dir resolve, KeyboardInterrupt, unhandled exception) + the success path; flattening would obscure the failure modes
    """Entry point."""
    args = parse_arguments()
    logger.setLevel(args.log_level)

    try:
        sim_config = load_simulation_config(args.config)
    except (FileNotFoundError, ValidationError) as exc:
        logger.exception("Failed to load config %s: %s", args.config, exc)
        return 1

    if sim_config.coevolution is None:
        logger.error(
            "Config %s does not define a `coevolution:` block. "
            "Use scripts/run_evolution.py for single-population "
            "campaigns or add a coevolution block to switch.",
            args.config,
        )
        return 1

    # Seed precedence: --seed > YAML top-level `seed:` > 0. Both the
    # master rng and the held-out rng inside CoevolutionLoop derive from
    # this value; downstream RNG-state divergence between runs with the
    # same `--seed` is a regression bug.
    seed: int = args.seed if args.seed is not None else (sim_config.seed or 0)
    rng = np.random.default_rng(seed)

    resolved = _resolve_session_dir(args.output_dir, args.resume)
    if resolved is None:
        return 1
    session_id, session_dir = resolved
    session_dir.mkdir(parents=True, exist_ok=True)

    log_path = configure_file_logging(session_id)
    if log_path is not None:
        logger.info("Log file: %s", log_path)

    cfg = sim_config.coevolution
    # Prominent startup logging mirrors run_evolution.py.
    logger.info("=" * 60)
    logger.info("CoevolutionLoop campaign")
    logger.info("Config:           %s", args.config)
    logger.info("Seed:             %d", seed)
    logger.info("Session dir:      %s", session_dir)
    logger.info("Resume:           %s", bool(args.resume))
    logger.info("K per block:      %d", cfg.K_per_block)
    logger.info(
        "Generation pairs: %d (= %d K-blocks)",
        cfg.generation_pairs,
        2 * cfg.generation_pairs,
    )
    logger.info("Probe cadence:    every %d gens", cfg.generality_probe_every)
    logger.info("Held-out size:    %d", cfg.held_out_size)
    logger.info("Start side:       %s", cfg.start_side)
    logger.info("Pred bootstrap:   %s", cfg.predator_gen0_bootstrap)
    logger.info(
        "Prey pop / pred pop: %d / %d",
        cfg.prey_evolution.population_size,
        cfg.predator_evolution.population_size,
    )
    logger.info("=" * 60)

    try:
        loop = CoevolutionLoop(
            sim_config=sim_config,
            output_dir=session_dir,
            rng=rng,
            log_level=getattr(logging, args.log_level),
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.exception("CoevolutionLoop construction failed: %s", exc)
        return 1

    try:
        loop.run(resume=bool(args.resume))
    except KeyboardInterrupt:
        logger.warning(
            "Run interrupted by user. The most recent K-block checkpoint "
            "(if any) is at %s — re-run with --resume %s to continue.",
            session_dir,
            session_dir,
        )
        return 130
    except Exception:  # noqa: BLE001 — campaign drivers must surface unhandled errors with full traceback
        logger.exception("CoevolutionLoop.run raised an unhandled exception")
        return 1

    logger.info("Co-evolution run complete. Artefacts at %s", session_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
