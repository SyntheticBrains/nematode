"""Fit tasks on single-thread worker processes, checkpointed so a stopped run resumes.

Before starting and before each batch the runner refuses to go on while a C.2 process (or any
process matching config.BUSY_PATTERN) is running. A task is a JSON-serialisable dict; its result
is appended to a JSON-lines checkpoint, with a held-out prediction saved beside it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from ceiling import config as C


# Tasks between two checks for reserved processes: 8 per worker at the registered width.
BATCH_TASKS = 8 * C.WORKERS


class Busy(RuntimeError):
    """A process the core budget reserves is running."""


def busy_processes(pattern: str | None = None) -> list[str]:
    pattern = pattern or C.BUSY_PATTERN
    own = {os.getpid(), os.getppid()}
    listing = subprocess.run(["ps", "-A", "-o", "pid=,command="], capture_output=True, text=True,
                             check=True).stdout
    found = []
    for line in listing.splitlines():
        pid, _, command = line.strip().partition(" ")
        if pid.isdigit() and int(pid) not in own and re.search(pattern, command):
            found.append(line.strip())
    return found


def refuse_if_busy(pattern: str | None = None) -> None:
    found = busy_processes(pattern)
    if found:
        raise Busy("refused: reserved processes are running:\n  " + "\n  ".join(found[:5]))


# ---- checkpoint -----------------------------------------------------------------------------------


def task_id(task: dict) -> str:
    return task["id"]


def _pred_name(tid: str) -> str:
    return hashlib.sha256(tid.encode()).hexdigest()[:32] + ".npy"


class Checkpoint:
    def __init__(self, run_dir: Path, stage: str) -> None:
        self.dir = Path(run_dir)
        self.path = self.dir / "checkpoints" / f"{stage}.jsonl"
        self.preds = self.dir / "preds"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.preds.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line torn by a stop
                self.done[record["id"]] = record

    def record(self, result: dict) -> None:
        result = dict(result)
        pred = result.pop("pred", None)
        if pred is not None:
            name = _pred_name(result["id"])
            np.save(self.preds / name, pred)
            result["pred_file"] = name
        with open(self.path, "a") as handle:
            handle.write(json.dumps(result, default=float) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.done[result["id"]] = result

    def prediction(self, tid: str) -> np.ndarray | None:
        name = self.done[tid].get("pred_file")
        return None if name is None else np.load(self.preds / name)


# ---- workers --------------------------------------------------------------------------------------


def _init_worker() -> None:
    here = str(C.HERE)
    if here not in sys.path:
        sys.path.insert(0, here)
    C.single_thread()


def run_tasks(tasks: list[dict], run_dir: Path, stage: str, execute, workers: int = C.WORKERS,
              batch: int | None = None, check_busy: bool = True) -> Checkpoint:
    """Run the tasks not yet in the stage's checkpoint; workers=0 runs in this process."""
    checkpoint = Checkpoint(run_dir, stage)
    todo = [t for t in tasks if task_id(t) not in checkpoint.done]
    if not todo:
        return checkpoint
    if check_busy:
        refuse_if_busy()
    batch = batch or BATCH_TASKS
    job = functools.partial(execute, str(run_dir))
    if workers == 0:
        for start in range(0, len(todo), batch):
            if check_busy and start:
                refuse_if_busy()
            for task in todo[start:start + batch]:
                checkpoint.record(job(task))
        return checkpoint
    context = mp.get_context("spawn")
    with context.Pool(workers, initializer=_init_worker) as pool:
        for start in range(0, len(todo), batch):
            if check_busy and start:
                refuse_if_busy()
            for result in pool.imap_unordered(job, todo[start:start + batch]):
                checkpoint.record(result)
    return checkpoint
