#!/usr/bin/env python
"""Report a running campaign's progress: finished, in flight, pending, and an ETA.

Campaigns here run for hours, and the runner's own progress lines only reach whatever the launch
redirected stdout to -- which is easy to lose track of, and gone entirely if the launch was not
redirected. This reads the campaign directory instead, so it works from a fresh shell and needs
nothing but the path.

**How "finished" is counted.** The runner writes ``<label>.exit`` beside each run's log once the
child has exited, holding its return code. A run is finished when that marker exists and failed when
its code is non-zero. **Log size is not used where markers exist**: stdout is buffered, but stderr is
not, so a warning printed at load time makes a log non-empty while the run is still going.

Campaigns launched before the runner wrote markers have none. For those the reader falls back to
"a non-empty log is a finished run" and **says so in its output**, because that signal miscounts any
run that writes to stderr before it ends.

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


def campaign_done(campaign: Path, total: int | None) -> bool:
    """Whether THIS campaign has finished, whatever else is running on the machine.

    With ``--total`` that is simply every run accounted for. Without it the reader cannot know how
    many runs remain unstarted, so it waits until this campaign has nothing in flight and no
    simulation is running anywhere -- conservative, but it never stops early.
    """
    s = survey(campaign, total)
    if total is not None:
        return int(s["finished"]) >= total  # type: ignore[arg-type]
    return s["in_flight"] == 0 and not any_runner_alive()


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
    # Scoped to THIS campaign by construction: every count reads this campaign's own files, never
    # the machine's process table, which would attribute another campaign's workers to this one.
    run_logs = list(logs.glob("*.log"))
    markers = {marker.stem: marker for marker in logs.glob("*.exit")}
    if markers:
        codes = [marker.read_text().strip() for marker in markers.values()]
        finished = len(markers)
        failed = sum(1 for code in codes if code != "0")
        running = sum(1 for log in run_logs if log.stem not in markers)
        basis = "completion markers"
    else:
        finished = sum(1 for log in run_logs if log.stat().st_size > 0)
        running = len(run_logs) - finished
        failed = None
        basis = "log size (no completion markers; a run that writes to stderr early is miscounted)"
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
        "failed": failed,
        "elapsed": elapsed,
        "tracebacks": broken,
        "basis": basis,
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
    if s["failed"] is not None:
        tail += f"   failed {s['failed']}"
    tail += f"   tracebacks {s['tracebacks']}"
    lines.append(tail)
    lines.append(f"  counted from {s['basis']}")
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
    if args.total is not None and args.total <= 0:
        ap.error("--total must be a positive run count")

    while True:
        print(report(args.campaign, args.total), flush=True)
        if not args.watch:
            return 0
        if campaign_done(args.campaign, args.total):
            print("  campaign has finished", flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
