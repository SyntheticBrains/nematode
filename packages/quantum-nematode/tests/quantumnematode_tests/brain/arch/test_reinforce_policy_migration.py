"""Migration regression for the REINFORCE brains onto ``_policy.py`` (Task 5).

**Family D** — `spikingreinforce`, `qrc`, `mlpreinforce`, and the
`hybridquantumcortex` cortex path. These share the scorers with the PPO families
but use ``reinforce_policy_loss`` instead of ``ppo_clip_policy_loss``.

The family splits on byte-exactness, and the split is asserted rather than
assumed:

- ``spikingreinforce`` and the ``hybridquantumcortex`` cortex path already wrote
  the vectorised ``-(log_probs * advantages).mean()``, so their migration is
  **byte-exact**.
- ``qrc`` and ``mlpreinforce`` accumulated the same quantity in a Python loop and
  divided at the end. Replacing that with torch's blocked sum reassociates the
  additions — a float32 reorder, so they are **not** byte-exact and fall under the
  declared ~1e-7 bar (design D5).
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import torch
from quantumnematode.brain.arch._policy import (
    categorical_logprob_entropy_from_probs,
    categorical_logprob_entropy_torch,
    reinforce_policy_loss,
)

_SEED = 8123
_REORDER_TOLERANCE = 1e-7


class TestByteExactMembers:
    """The two members whose pre-migration code was already vectorised."""

    def test_matches_the_vectorised_inline_expression(self) -> None:
        torch.manual_seed(_SEED)
        log_probs = torch.randn(64)
        advantages = torch.randn(64)

        assert torch.equal(
            reinforce_policy_loss(log_probs, advantages),
            -(log_probs * advantages).mean(),
        )

    def test_detached_advantages_form_matches(self) -> None:
        """The ``hybridquantumcortex`` cortex path detaches its advantages."""
        torch.manual_seed(_SEED)
        log_probs = torch.randn(32, requires_grad=True)
        advantages = torch.randn(32, requires_grad=True)

        assert torch.equal(
            reinforce_policy_loss(log_probs, advantages.detach()),
            -(log_probs * advantages.detach()).mean(),
        )

    def test_spikingreinforce_floored_distribution_needs_the_probs_scorer(self) -> None:
        """Its probability floor makes the distribution not a softmax of the logits.

        ``spikingreinforce`` applies ``_apply_probability_floor`` before scoring,
        so routing it through the logits scorer would score a different
        distribution — the same trap Family C sets.
        """
        torch.manual_seed(_SEED)
        logits = torch.randn(4) * 2.0
        probs = torch.softmax(logits, dim=-1)
        floored = torch.clamp(probs, min=0.05)
        floored = floored / floored.sum()
        action = int(torch.argmin(probs).item())

        via_probs, _e, _p = categorical_logprob_entropy_from_probs(floored, action)
        via_logits, _e, _p = categorical_logprob_entropy_torch(logits, action)

        # Byte-exact against the inline Categorical on the floored vector...
        assert torch.equal(
            via_probs,
            torch.distributions.Categorical(floored).log_prob(torch.tensor(action)),
        )
        # ...and materially different from scoring the unfloored logits.
        assert not torch.allclose(via_probs, via_logits, rtol=0, atol=1e-3)


class TestLoopAccumulatedMembers:
    """``qrc`` / ``mlpreinforce`` reassociate their sum, so they are not byte-exact."""

    @staticmethod
    def _loop_accumulated(log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """Reproduce the pre-migration Python-loop accumulation exactly."""
        acc = torch.tensor(0.0)
        for t in range(len(log_probs)):
            acc = acc - log_probs[t] * advantages[t]
        return acc / len(log_probs)

    def test_agrees_with_the_loop_within_the_declared_reorder_tolerance(self) -> None:
        torch.manual_seed(_SEED)
        for n in (8, 64, 256):
            log_probs = torch.randn(n)
            advantages = torch.randn(n)
            assert torch.allclose(
                reinforce_policy_loss(log_probs, advantages),
                self._loop_accumulated(log_probs, advantages),
                rtol=0,
                atol=_REORDER_TOLERANCE,
            )

    def test_the_difference_is_real_and_is_a_reorder_not_a_formula_change(self) -> None:
        """Confirms the deviation is float32 association, not different maths.

        In float64 the two forms agree far more tightly than in float32, which is
        the signature of a summation reorder rather than a changed expression.
        """
        torch.manual_seed(_SEED)
        log_probs = torch.randn(512)
        advantages = torch.randn(512)

        gap32 = abs(
            float(reinforce_policy_loss(log_probs, advantages))
            - float(self._loop_accumulated(log_probs, advantages)),
        )
        lp64, adv64 = log_probs.double(), advantages.double()
        gap64 = abs(
            float(reinforce_policy_loss(lp64, adv64)) - float(self._loop_accumulated(lp64, adv64)),
        )

        assert gap64 < gap32 or gap32 == 0.0

    def test_mlpreinforce_loss_decomposition_is_algebraically_exact(self) -> None:
        """``(policy + entropy) / T`` == ``mean(policy) - beta * mean(H)``.

        ``mlpreinforce`` folded its entropy bonus into the same division. The
        migration splits them into two terms; this pins that the split is an
        identity, so only the summation reorder is in play.
        """
        torch.manual_seed(_SEED)
        n, beta = 32, 0.01
        log_probs = torch.randn(n).double()
        advantages = torch.randn(n).double()
        entropies = torch.rand(n).double()

        policy_acc = torch.tensor(0.0, dtype=torch.float64)
        entropy_acc = torch.tensor(0.0, dtype=torch.float64)
        for t in range(n):
            policy_acc = policy_acc - log_probs[t] * advantages[t]
            entropy_acc = entropy_acc - beta * entropies[t]
        inline_total = (policy_acc + entropy_acc) / n

        migrated_total = reinforce_policy_loss(log_probs, advantages) - beta * entropies.mean()

        assert torch.allclose(inline_total, migrated_total, rtol=0, atol=1e-12)
