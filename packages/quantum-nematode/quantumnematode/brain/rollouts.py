"""Per-step rollout recording: what a brain saw and what it did, one JSON line per step.

A :class:`RolloutRecorder` is attached to an agent; the single-agent runners call
:meth:`RolloutRecorder.record` immediately after the brain returns its action, and the
simulation entry point calls :meth:`RolloutRecorder.end_episode` after each episode and
:meth:`RolloutRecorder.close` at session end. With no recorder attached nothing is called
and nothing is written.

Each line carries the episode index, the step index within it, the ``BrainParams`` the
brain read (unset fields and the embedded previous action dropped), the sampled continuous
action, the action mean the policy reported (``None`` on brains that do not) and the
sample's probability. The file is flushed at the end of every episode so a partial
recording is usable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quantumnematode.brain.actions import ActionData
    from quantumnematode.brain.arch import BrainParams


def params_record(params: BrainParams) -> dict[str, Any]:
    """Serialise ``BrainParams`` as the recorder writes them: unset fields and action dropped."""
    return params.model_dump(mode="json", exclude_none=True, exclude={"action"})


class RolloutRecorder:
    """Append one JSON line per step to ``path``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self.episode = 0
        self.step = 0
        self.lines = 0

    def record(self, params: BrainParams, action: ActionData) -> None:
        """Write the current step: the observation the brain read and the action it returned."""
        row = {
            "episode": self.episode,
            "step": self.step,
            "params": params_record(params),
            "action": list(action.continuous) if action.continuous is not None else None,
            "action_mean": (
                list(action.continuous_mean) if action.continuous_mean is not None else None
            ),
            "probability": action.probability,
        }
        self._handle.write(json.dumps(row) + "\n")
        self.step += 1
        self.lines += 1

    def end_episode(self) -> None:
        """Flush the episode's lines and advance the episode index."""
        self._handle.flush()
        self.episode += 1
        self.step = 0

    def close(self) -> None:
        """Flush and close; safe to call twice."""
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


def read_rollouts(path: Path) -> list[dict[str, Any]]:
    """Read every line of a recording back as dictionaries."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
