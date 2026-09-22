#!/usr/bin/env python
"""Report a running campaign's progress: finished, in flight, pending, and an ETA.

Campaigns here run for hours, and the runner's own progress lines only reach whatever the launch
redirected stdout to -- which is easy to lose track of, and gone entirely if the launch was not
redirected. This reads the campaign directory instead, so it works from a fresh shell and needs
nothing but the path.

**How "finished" is counted, and why it is reliable.** Each run writes one log under ``logs/``, and
Python buffers stdout when it is not a terminal, so a run's log stays **zero bytes until it
completes**. A non-empty log is therefore a finished run. This is not a heuristic about content: it
is a property of the redirect the runner sets up.

Usage::

    uv run python scripts/campaigns/campaign_progress.py --campaign campaigns/a2-reading
    uv run python scripts/campaigns/campaign_progress.py --campaign campaigns/a2-reading --watch
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


def any_runner_alive() -> bool:
    """Whether any simulation process is running at all, campaign or not."""
    found = subprocess.run(
        ["pgrep", "-f", "run_simulation.py"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return any(line.strip() for line in found.stdout.splitlines())


def survey(campaign: Path, total: int | None) -> dict[str, object]:
    """Read the campaign directory: what has finished, what is running, what is left."""
    logs = campaign / "logs"
    if not logs.is_dir():
        logs = campaign
    # Scoped to THIS campaign by construction. The runner opens a run's log when it starts the
    # run and the run writes nothing until it ends, so an empty log is a run in flight and a
    # non-empty one is a run finished. Counting live processes instead would attribute another
    # campaign's workers to this one, which is exactly what it did on first use.
    finished = sum(1 for log in logs.glob("*.log") if log.stat().st_size > 0)
    running = sum(1 for log in logs.glob("*.log") if log.stat().st_size == 0)
    # st_birthtime is macOS-only; st_ctime is the portable stand-in, and for a directory the
    # runner created and never moves it is the same instant in practice.
    stat = campaign.stat()
    started = datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_ctime), tz=UTC)
    elapsed = datetime.now(tz=UTC) - started
    # A traceback in a run's log is the failure signature worth surfacing: the runner reports a
    # non-zero exit per run, but that line lives in its stdout, not here.
    broken = sum(1 for log in logs.glob("*.log") if "Traceback" in log.read_text(errors="ignore"))
    out: dict[str, object] = {
        "finished": finished,
        "in_flight": running,
        "elapsed": elapsed,
        "tracebacks": broken,
    }
    if total is not None:
        out["pending"] = max(0, total - finished - running)
        out["total"] = total
        if finished:
            out["eta"] = timedelta(seconds=elapsed.total_seconds() * (total - finished) / finished)
    return out


def _fmt(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m"


def report(campaign: Path, total: int | None) -> str:
    """Render one progress line set."""
    s = survey(campaign, total)
    lines = [f"{campaign}"]
    if "total" in s:
        done, all_ = int(s["finished"]), int(s["total"])  # type: ignore[arg-type]
        lines.append(
            f"  finished {done}/{all_} ({done * 100 // all_}%)   "
            f"in flight {s['in_flight']}   pending {s['pending']}",
        )
    else:
        lines.append(f"  finished {s['finished']}   in flight {s['in_flight']}")
    elapsed = s["elapsed"]
    tail = f"  elapsed {_fmt(elapsed)}" if isinstance(elapsed, timedelta) else "  elapsed ?"
    eta = s.get("eta")
    if isinstance(eta, timedelta):
        done_at = (datetime.now(tz=UTC) + eta).astimezone()
        tail += f"   eta ~{_fmt(eta)} (about {done_at:%H:%M})"
    tail += f"   tracebacks {s['tracebacks']}"
    lines.append(tail)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: print progress once, or every ``--interval`` seconds under ``--watch``."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--campaign", type=Path, required=True, help="campaign directory")
    ap.add_argument("--total", type=int, help="expected run count, for a percentage and an ETA")
    ap.add_argument("--watch", action="store_true", help="reprint until the campaign finishes")
    ap.add_argument("--interval", type=int, default=120, help="seconds between reprints")
    args = ap.parse_args(argv)

    if not args.campaign.is_dir():
        print(f"no such campaign directory: {args.campaign}", file=sys.stderr)
        return 2

    while True:
        print(report(args.campaign, args.total), flush=True)
        if not args.watch:
            return 0
        if not any_runner_alive():
            print("  runner has exited", flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
