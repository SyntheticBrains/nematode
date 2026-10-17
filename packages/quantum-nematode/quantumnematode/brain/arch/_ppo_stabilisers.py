"""PPO stabilisers shared by every PPO learner, each off by default and inert when off.

Small PPO learners trained long enough can lose a policy they had. These settings are the standard
candidate remedies, applied identically to every learner that carries them:

* **KL early stopping, relative to the learner** (``kl_stop_factor``): before each minibatch's
  step, if that minibatch's approximate KL exceeds the factor times the median, over the learner's
  last 100 updates, of each update's largest minibatch KL, the step is not taken and the rest of the
  update is skipped. The window holds the same statistic the test compares, a single minibatch's KL,
  so the stop acts only on minibatches far beyond what the learner's updates usually reach. A fixed
  KL target cannot serve learners whose typical KL differs tenfold.
* **Weight clipping relative to the initial scale** (``weight_clip``, a factor κ): after every
  optimiser step each learnable tensor is clipped elementwise to ±κ times its own largest absolute
  value at construction. A tensor that is zero everywhere at construction (a bias, a log-std, a
  log-gain) has no scale to be relative to and is left unclipped.
* **Adam's second-moment settings** (``adam_eps``, ``adam_beta2``), passed to the optimiser and
  re-applied after an optimiser state is loaded, which would otherwise restore the saved ones.
* **The placebo** (``rollout_shuffle_stream``): the rollout buffer draws its minibatch permutations
  from a generator keyed on the run seed and a fixed tag. It re-rolls a run's trajectory with no
  stabilising intent, so a remedy can be read against a re-roll rather than against the run it was
  chosen to fix.
"""

from __future__ import annotations

from collections import deque
from statistics import median
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import optim

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn

# The window the KL stop's running median is taken over, in updates; no stop before it fills.
KL_STOP_WINDOW = 100
# Tag appended to the run seed for the placebo's permutation stream.
_SHUFFLE_STREAM_TAG = 0x504C_4342
# PyTorch's Adam defaults, kept wherever only one of the two settings is given.
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPS = 1e-8


class PPOStabiliserConfig(BaseModel):
    """The stabilisers' configuration keys; every one off by default."""

    kl_stop_factor: float | None = Field(default=None, gt=0.0)
    weight_clip: float | None = Field(default=None, gt=0.0)
    adam_eps: float | None = Field(default=None, gt=0.0)
    adam_beta2: float | None = Field(default=None, gt=0.0, lt=1.0)
    rollout_shuffle_stream: bool = False


def make_adam(
    params: Iterable[nn.Parameter],
    learning_rate: float,
    config: PPOStabiliserConfig,
) -> optim.Adam:
    """Build the optimiser: PyTorch's defaults unless the second-moment settings are given."""
    if config.adam_eps is None and config.adam_beta2 is None:
        return optim.Adam(params, lr=learning_rate)
    return optim.Adam(
        params,
        lr=learning_rate,
        betas=(_ADAM_BETA1, config.adam_beta2 if config.adam_beta2 is not None else _ADAM_BETA2),
        eps=config.adam_eps if config.adam_eps is not None else _ADAM_EPS,
    )


def reapply_adam(optimizer: optim.Optimizer, config: PPOStabiliserConfig) -> None:
    """Restore the configured second-moment settings after an optimiser state is loaded."""
    for group in optimizer.param_groups:
        if config.adam_eps is not None:
            group["eps"] = config.adam_eps
        if config.adam_beta2 is not None:
            group["betas"] = (group["betas"][0], config.adam_beta2)


class WeightClip:
    """Clip each tensor to ±κ times its largest magnitude at construction, after every step."""

    def __init__(self, params: Iterable[nn.Parameter], factor: float) -> None:
        self.bounds: list[tuple[nn.Parameter, float]] = []
        for param in params:
            scale = float(param.detach().abs().max()) if param.numel() else 0.0
            if scale > 0.0:
                self.bounds.append((param, factor * scale))

    def apply(self) -> None:
        """Clip in place, outside the autograd graph."""
        with torch.no_grad():
            for param, bound in self.bounds:
                param.clamp_(-bound, bound)


class KLStop:
    """Stop an update whose next minibatch would move the policy far more than the learner's norm.

    The window holds each update's largest minibatch KL, the statistic each minibatch is tested on.
    """

    def __init__(self, factor: float, window: int = KL_STOP_WINDOW) -> None:
        self.factor = factor
        self.history: deque[float] = deque(maxlen=window)

    def exceeded(self, minibatch_kl: float) -> bool:
        """Whether a minibatch's KL calls for the stop; never before the window has filled."""
        if len(self.history) < (self.history.maxlen or 0):
            return False
        return minibatch_kl > self.factor * median(self.history)

    def record(self, update_kl: float) -> None:
        """Add one update's KL to the window."""
        self.history.append(update_kl)


class UpdateGate:
    """One update's KL stop: whether each minibatch may step, and the update's KL once it ends.

    Without a KL stop every minibatch steps and nothing is computed.
    """

    def __init__(self, kl_stop: KLStop | None) -> None:
        self._stop = kl_stop
        self._kls: list[float] = []
        self.stopped = False
        self.stepped = 0

    def allow(self, new_log_probs: torch.Tensor, old_log_probs: torch.Tensor) -> bool:
        """Whether this minibatch may take its step; once refused, the update is over."""
        if self._stop is not None:
            kl = approx_kl(new_log_probs, old_log_probs)
            self._kls.append(kl)
            if self._stop.exceeded(kl):
                self.stopped = True
                return False
        self.stepped += 1
        return True

    def close(self) -> None:
        """Record the update's largest minibatch KL in the stop's window."""
        if self._stop is not None and self._kls:
            self._stop.record(max(self._kls))


def approx_kl(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor) -> float:
    """Return the low-variance KL estimate ``mean((r - 1) - log r)``, r the probability ratio."""
    log_ratio = (new_log_probs - old_log_probs).detach().double()
    return float(((log_ratio.exp() - 1.0) - log_ratio).mean())


def shuffle_rng(seed: int) -> np.random.Generator:
    """Return the placebo's permutation generator for a run seed."""
    return np.random.default_rng([seed, _SHUFFLE_STREAM_TAG])
