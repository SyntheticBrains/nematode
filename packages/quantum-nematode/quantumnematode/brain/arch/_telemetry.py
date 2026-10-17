"""Per-update PPO telemetry: statistics an update already holds, written one JSON line per update.

Everything here reads tensors and writes a file. Nothing draws from a random generator, calls a
sampler, or writes to a parameter, a buffer or an optimiser, so a run with telemetry on is the run
with it off: the same episodes, the same weights and the same next value of every generator.

What each line carries, and why each is there: the late-training collapse of small PPO learners has
several documented mechanisms, and each leaves a different trace before the success rate falls.

* weight growth driving saturation and loss of rank: layer-wise weight norms, the saturated-unit
  fraction and the effective rank (``srank``) of the features the heads read;
* a fixed step against a vanishing signal: the raw advantages' spread before normalisation, beside
  approximate KL, clip fraction and the gradient norm;
* stale optimiser second moments: a jump in the pre-clip gradient norm followed by a KL spike;
* critic failure: explained variance and value loss, and the critic's gradient norm beside the
  actor's;
* exploration collapse: ``log_std``;
* a saturating mean head: the pre-squash action mean's magnitude and the fraction of steps with
  ``|u| > 2``.

On the connectome, the settling residual says how far the settled state is from a fixed point.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from torch import nn

# A tanh unit at or beyond this magnitude passes almost no gradient.
TANH_SATURATED = 0.99
# A pre-squash mean beyond this sits where tanh's slope is below 0.08 of its maximum.
LARGE_MEAN = 2.0
# ``srank``: the fewest singular values whose sum reaches this fraction of the total.
SRANK_KEEP = 0.99
# ``srank`` grows with the number of rows it is computed on, and an update's batch is as long as the
# episode that ended it, so it is computed on this many rows, the first of the final epoch's
# minibatches in the order the update drew them. A smaller batch reports its own row count.
SRANK_ROWS = 64


class UpdateTelemetry:
    """Append one JSON line per PPO update to ``path``.

    ``episode`` is set by the runner before each episode, and ``update`` counts the lines written.
    The file is opened on the first write, so a run that makes no update writes nothing.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.episode = 0
        self.update = 0
        self._handle: Any = None

    def write(self, record: dict[str, Any]) -> None:
        """Write one update's record, stamped with the episode and the update index."""
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a")
        line = {"episode": self.episode, "update": self.update, **record}
        self._handle.write(json.dumps(line, sort_keys=True) + "\n")
        self._handle.flush()
        self.update += 1

    def close(self) -> None:
        """Close the file if one was opened."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def grad_norm(params: Iterable[nn.Parameter]) -> float:
    """L2 norm of the gradients held by ``params``, before any clipping."""
    squares = [p.grad.detach().double().pow(2).sum() for p in params if p.grad is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def approx_kl(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor) -> float:
    """Return the low-variance KL estimate ``mean((r - 1) - log r)``, r the probability ratio."""
    log_ratio = (new_log_probs - old_log_probs).detach().double()
    return float(((log_ratio.exp() - 1.0) - log_ratio).mean())


def clip_fraction(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_epsilon: float,
) -> float:
    """Fraction of samples whose probability ratio left ``[1 - eps, 1 + eps]``."""
    ratio = (new_log_probs - old_log_probs).detach().double().exp()
    return float(((ratio - 1.0).abs() > clip_epsilon).double().mean())


def explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float | None:
    """``1 - Var(returns - values) / Var(returns)``; ``None`` when the returns do not vary."""
    returns, values = returns.detach().double(), values.detach().double()
    spread = returns.var(unbiased=False)
    if float(spread) == 0.0:
        return None
    return float(1.0 - (returns - values).var(unbiased=False) / spread)


def srank(features: torch.Tensor, keep: float = SRANK_KEEP) -> int:
    """Count the fewest singular values of ``features`` (rows are samples) reaching ``keep``."""
    values = torch.linalg.svdvals(features.detach().double())
    total = float(values.sum())
    if total == 0.0:
        return 0
    cumulative = torch.cumsum(values, dim=0) / total
    return int((cumulative < keep).sum()) + 1


def saturated_fraction(features: torch.Tensor, activation: str) -> float:
    """Fraction of feature entries passing almost no gradient.

    For tanh, ``|h| > 0.99``; for ReLU, exactly zero (an inactive unit passes none).
    """
    detached = features.detach()
    if activation == "relu":
        return float((detached == 0).double().mean())
    return float((detached.abs() > TANH_SATURATED).double().mean())


def weight_norms(modules: dict[str, nn.Module | nn.Parameter]) -> dict[str, float]:
    """Frobenius norm of every named parameter, prefixed by the module it belongs to."""
    out: dict[str, float] = {}
    for prefix, module in modules.items():
        if isinstance(module, torch.nn.Parameter):
            out[prefix] = float(module.detach().double().norm())
            continue
        for name, param in module.named_parameters():
            out[f"{prefix}.{name}"] = float(param.detach().double().norm())
    return out


class UpdateRecord:
    """One update's statistics, gathered minibatch by minibatch.

    Entropy, losses and gradient norms are means over every minibatch of every epoch, as the
    update's own report is. KL, clip fraction, the action mean and the features are read over the
    final epoch, whose weights the next rollout uses: ``begin_epoch`` clears them.
    """

    def __init__(
        self,
        *,
        returns: torch.Tensor,
        raw_advantages: torch.Tensor,
        old_values: torch.Tensor,
        clip_epsilon: float,
        max_grad_norm: float,
    ) -> None:
        raw = raw_advantages.detach().double()
        self._fixed: dict[str, Any] = {
            "adv_mean": float(raw.mean()),
            "adv_std": float(raw.std(unbiased=False)),
            "adv_max_abs": float(raw.abs().max()),
            "explained_variance": explained_variance(returns, old_values),
            "batch_size": len(raw),
        }
        self._clip_epsilon = clip_epsilon
        self._max_grad_norm = max_grad_norm
        self._all: dict[str, list[float]] = {
            k: [] for k in ("entropy", "value_loss", "grad_norm", "actor_grad", "critic_grad")
        }
        self._final: dict[str, list[torch.Tensor]] = {}
        self._grads: tuple[float, float] = (0.0, 0.0)

    def read_grads(self, actor: Iterable[nn.Parameter], critic: Iterable[nn.Parameter]) -> None:
        """Read the actor's and the critic's gradient norms, after backward and before clipping."""
        self._grads = (grad_norm(actor), grad_norm(critic))

    def begin_epoch(self) -> None:
        """Start an epoch: the final-epoch statistics keep only the latest one."""
        self._final = {"new": [], "old": [], "mean": [], "features": [], "residual": []}

    def minibatch(  # noqa: PLR0913 - one argument per statistic the minibatch holds
        self,
        *,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        head_out: torch.Tensor,
        features: torch.Tensor,
        entropy: float,
        value_loss: float,
        grad_total: float,
        penultimate: torch.Tensor | None = None,
    ) -> None:
        """Record one minibatch, with the gradient norms ``read_grads`` took for it."""
        for key, value in (
            ("entropy", entropy),
            ("value_loss", value_loss),
            ("grad_norm", grad_total),
            ("actor_grad", self._grads[0]),
            ("critic_grad", self._grads[1]),
        ):
            self._all[key].append(float(value))
        self._final["new"].append(new_log_probs.detach())
        self._final["old"].append(old_log_probs.detach())
        self._final["mean"].append(head_out.detach())
        self._final["features"].append(features.detach())
        if penultimate is not None:
            self._final["residual"].append((features.detach() - penultimate).norm(dim=-1))

    def finish(
        self,
        *,
        log_std: torch.Tensor | None,
        norms: dict[str, float],
        activation: str,
        continuous: bool,
    ) -> dict[str, Any]:
        """Assemble the update's line."""
        new, old = torch.cat(self._final["new"]), torch.cat(self._final["old"])
        features = torch.cat(self._final["features"])
        grads = self._all["grad_norm"]
        out: dict[str, Any] = {
            **self._fixed,
            **{k: _mean(v) for k, v in self._all.items()},
            "clipped_minibatch_fraction": (
                sum(1 for g in grads if g > self._max_grad_norm) / len(grads) if grads else None
            ),
            "approx_kl": approx_kl(new, old),
            "clip_fraction": clip_fraction(new, old, self._clip_epsilon),
            "saturated_fraction": saturated_fraction(features, activation),
            "saturation_rule": "relu_zero" if activation == "relu" else "tanh_0.99",
            "srank": srank(features[:SRANK_ROWS]),
            "srank_rows": int(min(SRANK_ROWS, features.shape[0])),
            "feature_dim": int(features.shape[-1]),
            "weight_norms": norms,
            "log_std": None if log_std is None else float(log_std.detach().double().mean()),
        }
        if continuous:
            mean = torch.cat(self._final["mean"]).detach().double()
            out["action_mean_abs"] = float(mean.abs().mean())
            # A step counts once, however many of its action dimensions are large.
            large = (
                mean.abs().amax(dim=-1) > LARGE_MEAN if mean.ndim > 1 else mean.abs() > LARGE_MEAN
            )
            out["action_mean_large_fraction"] = float(large.double().mean())
        if self._final["residual"]:
            residual = torch.cat(self._final["residual"]).double()
            out["settling_residual"] = float(residual.mean())
            out["settling_residual_max"] = float(residual.max())
        return out


class NoRecord(UpdateRecord):
    """The record an update keeps with telemetry off: every call does nothing."""

    def __init__(self) -> None:
        """Hold nothing."""

    def read_grads(self, actor: Iterable[nn.Parameter], critic: Iterable[nn.Parameter]) -> None:
        """Read nothing."""

    def begin_epoch(self) -> None:
        """Keep nothing."""

    def minibatch(self, **_: object) -> None:  # type: ignore[override]
        """Keep nothing."""


def record_for(  # noqa: PLR0913 - the record's inputs, and the switch between the two records
    telemetry: UpdateTelemetry | None,
    *,
    returns: torch.Tensor,
    raw_advantages: torch.Tensor,
    old_values: Iterable[torch.Tensor],
    clip_epsilon: float,
    max_grad_norm: float,
) -> UpdateRecord:
    """Return a live record when telemetry is on, and the no-op record when it is off."""
    if telemetry is None:
        return NoRecord()
    return UpdateRecord(
        returns=returns,
        raw_advantages=raw_advantages,
        old_values=torch.stack(list(old_values)).reshape(-1),
        clip_epsilon=clip_epsilon,
        max_grad_norm=max_grad_norm,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
