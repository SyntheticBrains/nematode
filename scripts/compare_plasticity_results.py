#!/usr/bin/env python3
"""Quantum Plasticity Test — Cross-architecture comparison.

Reads aggregate CSVs from multiple plasticity test runs and computes
forgetting ratios and t-test p-values for quantum vs classical pairs.

Pairs: QRH vs CRH, HybridQuantum vs HybridClassical.
MLP PPO serves as overall classical baseline.

Usage:
    uv run python scripts/compare_plasticity_results.py \
        --results exports/run1/plasticity/aggregate_metrics.csv \
                  exports/run2/plasticity/aggregate_metrics.csv \
                  exports/run3/plasticity/aggregate_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.stats import ttest_ind

console = Console()

# Quantum → Classical control pairs
QUANTUM_CLASSICAL_PAIRS: list[tuple[str, str]] = [
    ("qrh", "crh"),
    ("hybridquantum", "hybridclassical"),
]


@dataclass
class ArchMetrics:
    """Aggregate metrics for a single architecture."""

    name: str
    bf_values: list[float]
    ft_values: list[float]
    pr_values: list[float]
    bf_mean: float = 0.0
    bf_std: float = 0.0


def load_aggregate_csv(path: Path) -> ArchMetrics | None:
    """Load an aggregate_metrics.csv and extract BF/FT/PR values."""
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        return None

    name = ""
    bf_values: list[float] = []
    ft_values: list[float] = []
    pr_values: list[float] = []

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["architecture"]
            values = [float(v) for v in row["values"].split(";")]
            if row["metric"] == "backward_forgetting":
                bf_values = values
            elif row["metric"] == "forward_transfer":
                ft_values = values
            elif row["metric"] == "plasticity_retention":
                pr_values = values

    if not name:
        console.print(f"[red]No data found in: {path}[/red]")
        return None

    metrics = ArchMetrics(
        name=name,
        bf_values=bf_values,
        ft_values=ft_values,
        pr_values=pr_values,
    )
    if bf_values:
        metrics.bf_mean = float(np.mean(bf_values))
        metrics.bf_std = float(np.std(bf_values))
    return metrics


def main() -> None:
    """Entry point for the comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare plasticity test results across architectures",
    )
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help="Paths to aggregate_metrics.csv files from plasticity test runs",
    )
    args = parser.parse_args()

    # Load all results
    all_metrics: dict[str, ArchMetrics] = {}
    for path_str in args.results:
        path = Path(path_str)
        metrics = load_aggregate_csv(path)
        if metrics:
            all_metrics[metrics.name] = metrics
            console.print(f"Loaded: {metrics.name} from {path}")

    if not all_metrics:
        console.print("[red]No valid results loaded.[/red]")
        return

    # Summary table
    summary_table = Table(title="Plasticity Results Summary")
    summary_table.add_column("Architecture", style="cyan")
    summary_table.add_column("BF (mean±std)", justify="right")
    summary_table.add_column("FT (mean±std)", justify="right")
    summary_table.add_column("PR (mean±std)", justify="right")
    summary_table.add_column("N seeds", justify="right")

    for name, m in sorted(all_metrics.items()):
        bf_str = f"{m.bf_mean:.4f}±{m.bf_std:.4f}" if m.bf_values else "N/A"
        ft_str = f"{np.mean(m.ft_values):.4f}±{np.std(m.ft_values):.4f}" if m.ft_values else "N/A"
        pr_str = f"{np.mean(m.pr_values):.4f}±{np.std(m.pr_values):.4f}" if m.pr_values else "N/A"
        n = max(len(m.bf_values), len(m.ft_values), len(m.pr_values))
        summary_table.add_row(name, bf_str, ft_str, pr_str, str(n))

    console.print(summary_table)

    _print_comparison_table(all_metrics)


def _print_comparison_table(all_metrics: dict[str, ArchMetrics]) -> None:
    """Print quantum vs classical forgetting comparison table."""
    comparison_table = Table(title="Quantum vs Classical Forgetting Comparison")
    comparison_table.add_column("Pair", style="cyan")
    comparison_table.add_column("Q BF", justify="right")
    comparison_table.add_column("C BF", justify="right")
    comparison_table.add_column("FR (Q/C)", justify="right")
    comparison_table.add_column("p-value", justify="right")
    comparison_table.add_column("Verdict", style="bold")

    for q_name, c_name in QUANTUM_CLASSICAL_PAIRS:
        q = all_metrics.get(q_name)
        c = all_metrics.get(c_name)
        if not q or not c:
            comparison_table.add_row(
                f"{q_name} vs {c_name}",
                "—",
                "—",
                "—",
                "—",
                "[dim]Missing data[/dim]",
            )
            continue

        if not q.bf_values or not c.bf_values:
            comparison_table.add_row(
                f"{q_name} vs {c_name}",
                "—",
                "—",
                "—",
                "—",
                "[dim]No BF data[/dim]",
            )
            continue

        fr = q.bf_mean / c.bf_mean if c.bf_mean != 0 else float("inf")
        ttest_result = ttest_ind(q.bf_values, c.bf_values)
        p_value: float = ttest_result[1]  # type: ignore[assignment]

        if fr <= 0.5 and p_value < 0.05:
            verdict = "[green]CONFIRMED[/green]"
        elif p_value < 0.05:
            verdict = f"[yellow]Significant (FR={fr:.2f})[/yellow]"
        else:
            verdict = f"[red]Not significant (p={p_value:.3f})[/red]"

        comparison_table.add_row(
            f"{q_name} vs {c_name}",
            f"{q.bf_mean:.4f}",
            f"{c.bf_mean:.4f}",
            f"{fr:.4f}",
            f"{p_value:.4f}",
            verdict,
        )

    console.print(comparison_table)
    console.print(
        "\n[dim]Hypothesis: FR ≤ 0.5 with p < 0.05 confirms quantum plasticity advantage[/dim]",
    )


if __name__ == "__main__":
    main()
