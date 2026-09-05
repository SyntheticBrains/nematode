#!/usr/bin/env python
"""Run the L4 panel's pilot: the recipe grid on the pilot seeds, through the campaign runner.

Derives one config per rule-bearing arm per ``plasticity_rate`` on the grid from the
committed arm configs -- the parent plus that one explicit key -- and writes them under
``<out>/configs/`` so the pilot is reproducible from its own directory. The frozen arms
run once from their committed configs, since their weights never move. The whole plan is
handed to ``scripts/run_campaign.py`` with experiment tracking on and rendering off, so
every run is the standard single-run entry point.

Usage::

    uv run python scripts/campaigns/l4_panel_pilot.py --out campaigns/l4-pilot [--dry-run]

    # extend one arm at the selected rate, a fresh run at the longer budget:
    uv run python scripts/campaigns/l4_panel_pilot.py --out campaigns/l4-pilot
        --only wt_plastic --rate 0.01 --runs 6000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import l4_panel  # noqa: E402  # pyright: ignore[reportMissingImports]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_CAMPAIGN = PROJECT_ROOT / "scripts" / "run_campaign.py"
PASSTHROUGH: tuple[str, ...] = ("--theme", "headless", "--track-experiment")


def derive_config(arm: str, rate: float, out_dir: Path) -> Path:
    """Write ``<stem>__rate_<enc>.yml``: the committed arm config plus one explicit rate."""
    stem = l4_panel.STEM_OF[arm]
    parent = l4_panel.CONFIG_DIR / f"{stem}.yml"
    data = yaml.safe_load(parent.read_text())
    data["brain"]["config"]["plasticity_rate"] = rate
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{stem}__rate_{l4_panel.encode_rate(rate)}.yml"
    header = (
        f"# Pilot grid point: {parent.name} with plasticity_rate {rate:g}.\n"
        "# Generated from the committed arm config; the only difference is that one key.\n"
    )
    target.write_text(header + yaml.safe_dump(data, sort_keys=False))
    return target


def plan_configs(
    out_dir: Path,
    *,
    arms: tuple[str, ...] = l4_panel.ARM_KEYS,
    rates: tuple[float, ...] = l4_panel.RATE_GRID,
) -> list[Path]:
    """Every config the pilot runs: derived grid points, then the committed frozen arms."""
    configs = [
        derive_config(arm, rate, out_dir / "configs")
        for arm in arms
        if arm in l4_panel.RULE_BEARING_ARMS
        for rate in rates
    ]
    configs += [
        l4_panel.CONFIG_DIR / f"{l4_panel.STEM_OF[arm]}.yml"
        for arm in arms
        if arm in l4_panel.FROZEN_ARMS
    ]
    return configs


def campaign_argv(configs: list[Path], args: argparse.Namespace) -> list[str]:
    """Build the campaign runner's argument list for these configs."""
    argv = [str(RUN_CAMPAIGN)]
    for config in configs:
        argv += ["--config", str(config)]
    argv += ["--seeds", args.seeds, "--runs", str(args.runs), "--output-dir", str(args.out)]
    if args.workers is not None:
        argv += ["--workers", str(args.workers)]
    if args.dry_run:
        argv.append("--dry-run")
    return [*argv, "--", *PASSTHROUGH]


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the pilot's own options."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, required=True, help="campaign output dir")
    ap.add_argument(
        "--seeds",
        default="-".join(str(s) for s in (l4_panel.PILOT_SEEDS[0], l4_panel.PILOT_SEEDS[-1])),
    )
    ap.add_argument("--runs", type=int, default=l4_panel.PILOT_BUDGET, help="episodes per run")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--only",
        action="append",
        choices=l4_panel.ARM_KEYS,
        default=None,
        help="restrict to these arms (repeatable)",
    )
    ap.add_argument(
        "--rate",
        action="append",
        type=float,
        choices=l4_panel.RATE_GRID,
        default=None,
        help="restrict to these grid rates (repeatable)",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Derive the grid configs and hand the plan to the campaign runner; return its exit code."""
    args = parse_arguments(list(sys.argv[1:]) if argv is None else argv)
    arms = tuple(args.only) if args.only else l4_panel.ARM_KEYS
    rates = tuple(args.rate) if args.rate else l4_panel.RATE_GRID
    configs = plan_configs(args.out, arms=arms, rates=rates)
    command = [sys.executable, *campaign_argv(configs, args)]
    print(f"{len(configs)} configs -> {RUN_CAMPAIGN.name}", flush=True)
    return subprocess.run(command, check=False).returncode  # noqa: S603


if __name__ == "__main__":
    sys.exit(main())
