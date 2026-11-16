#!/usr/bin/env python
"""Launch record 135's run, in its registered order. Nothing here runs on import.

    .venv/bin/python investigation/neural-ceiling/run.py setup     --run-dir <dir>
    .venv/bin/python investigation/neural-ceiling/run.py preflight --run-dir <dir> --confirm-registered-read
    .venv/bin/python investigation/neural-ceiling/run.py oracles   --run-dir <dir> --confirm-registered-read
    .venv/bin/python investigation/neural-ceiling/run.py real      --run-dir <dir>
    .venv/bin/python investigation/neural-ceiling/run.py score     --run-dir <dir> --confirm-registered-read
    .venv/bin/python investigation/neural-ceiling/run.py plan      (fit counts only; no data)

preflight reads S's finite pattern and oracles reads section 3's summaries, both through the
guarded scorer; score makes the registered reading. Each read refuses while the code under
ceiling/ has an uncommitted change, is logged in READS.md before S is decoded, and commits
READS.md (alone) after it. preflight also runs amendment 1's convergence check, which may raise
the cap of sparse-support and tied fits to 2,000 for every later stage. Every stage that fits refuses to start, and
stops before a batch, while a process matching c2_fit, c2_encoder_probe, run_campaign or
benchmark_grade is running. Fits are checkpointed in the run directory, so a stopped stage resumes
where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ceiling import config as C  # noqa: E402

C.single_thread()

from ceiling import plan as P  # noqa: E402
from ceiling import stages  # noqa: E402
from ceiling.runner import Busy, refuse_if_busy  # noqa: E402
from ceiling.scorer import Refusal  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["setup", "preflight", "oracles", "real", "score", "plan"])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--workers", type=int, default=C.WORKERS)
    parser.add_argument("--confirm-registered-read", action="store_true")
    parser.add_argument("--amendment", help="a committed amendment, for a repeat registered reading")
    args = parser.parse_args(argv)
    if args.workers > C.WORKERS:
        parser.error(f"at most {C.WORKERS} workers")
    if args.stage == "plan":
        print(json.dumps(P.fit_counts(), indent=1))
        return 0
    if args.run_dir is None:
        parser.error("--run-dir is required")
    try:
        if args.stage == "setup":
            refuse_if_busy()
            out = stages.setup(args.run_dir)
        elif args.stage == "preflight":
            out = stages.preflight(args.run_dir, args.confirm_registered_read, args.workers)
        elif args.stage == "oracles":
            out = stages.oracles_stage(args.run_dir, args.confirm_registered_read, args.workers)
        elif args.stage == "real":
            out = stages.real_stage(args.run_dir, args.workers)
        else:
            out = stages.score_stage(args.run_dir, args.confirm_registered_read, args.amendment)
            out = {"branches": out.get("branches")}
    except (Busy, Refusal) as exc:
        print(exc, file=sys.stderr)
        return 3
    print(json.dumps(out, indent=1, default=float)[:20000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
