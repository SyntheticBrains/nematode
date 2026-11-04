"""Slow speed noise on the encoded connectome brain.

Covers the connectome-encoder requirement "Slow speed noise behind a byte-identical default": off is
byte-identical, on appends one parameter, the process is a stationary OU on speed alone, PPO
evaluates the exact conditional, tau is held under PPO, and unsupported settings are refused.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import _policy as pol
from quantumnematode.brain.arch._ppo_buffer import PreviousStep, RolloutBuffer
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome import encoder as enc

from .test_connectome_encoder import _brain, _config, _params, _vector

_LOW = torch.tensor([-1.0] * 5)
_HIGH = torch.tensor([1.0] * 5)
_SLOW = {"encoder_speed_noise": "ou"}


def _sample_run(
    means: torch.Tensor,
    log_std: torch.Tensor,
    rho: float,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Sample a run of slow-noise actions: each step's pre-squash sample, log-prob and noise."""
    torch.manual_seed(seed)
    previous: torch.Tensor | None = None
    pres, log_probs, noises = [], [], []
    for mean in means:
        _, log_prob, _, pre, noise = pol.continuous_sample_tanh_gaussian_ou(
            mean,
            log_std,
            _LOW,
            _HIGH,
            previous,
            rho,
        )
        pres.append(pre)
        log_probs.append(log_prob)
        noises.append(noise)
        previous = noise
    return pres, log_probs, noises


class TestTheProcess:
    """Scenario: The process is a stationary OU on speed alone."""

    def test_it_is_stationary_with_lag_one_correlation_rho(self) -> None:
        means = torch.zeros(20_000, 5)
        _, _, noises = _sample_run(means, torch.zeros(5), 0.9, seed=1)
        x = torch.stack(noises).numpy()
        assert x.var() == pytest.approx(1.0, abs=0.08)
        assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(0.9, abs=0.01)

    def test_the_first_step_and_the_other_outputs_equal_the_independent_sampler(self) -> None:
        mean = torch.tensor([0.3, -0.2, 0.1, 0.0, -0.4])
        log_std = torch.tensor([-0.5, -1.0, 0.2, -2.0, 0.0])
        torch.manual_seed(7)
        action, log_prob, _, pre = pol.continuous_sample_tanh_gaussian(mean, log_std, _LOW, _HIGH)
        torch.manual_seed(7)
        slow = pol.continuous_sample_tanh_gaussian_ou(mean, log_std, _LOW, _HIGH, None, 0.9)
        assert torch.equal(slow[0], action)
        assert torch.equal(slow[1], log_prob)
        assert torch.equal(slow[3], pre)
        torch.manual_seed(7)
        later = pol.continuous_sample_tanh_gaussian_ou(
            mean,
            log_std,
            _LOW,
            _HIGH,
            torch.tensor(1.5),
            0.9,
        )
        assert torch.equal(later[3][1:], pre[1:])
        assert not torch.equal(later[3][0], pre[0])


class TestThePPODensity:
    """Scenario: PPO evaluates the exact conditional."""

    @staticmethod
    def _run() -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        means = torch.randn(60, 5, generator=torch.Generator().manual_seed(3)) * 0.5
        log_std = torch.tensor([-0.3, -1.0, -0.5, -0.5, -0.5])
        pres, log_probs, _ = _sample_run(means, log_std, 0.85, seed=4)
        return means, log_std, pres, log_probs

    @staticmethod
    def _evaluate(
        means: torch.Tensor,
        log_std: torch.Tensor,
        pres: list[torch.Tensor],
    ) -> torch.Tensor:
        pre = torch.stack(pres)
        live = torch.arange(len(pres)) > 0
        previous_mean = torch.cat([torch.zeros(1), means[:-1, 0]])
        previous_speed = torch.cat([torch.zeros(1), pre[:-1, 0]])
        log_probs, _ = pol.continuous_evaluate_tanh_gaussian_ou(
            means,
            log_std,
            pre,
            _LOW,
            _HIGH,
            previous_mean,
            previous_speed,
            live,
            0.85,
        )
        return log_probs

    def test_at_the_sampling_parameters_it_returns_the_samplers_log_prob(self) -> None:
        means, log_std, pres, log_probs = self._run()
        evaluated = self._evaluate(means, log_std, pres)
        assert torch.allclose(evaluated, torch.stack(log_probs), atol=1e-5)

    def test_under_new_parameters_it_is_the_exact_conditional(self) -> None:
        means, log_std, pres, _ = self._run()
        moved = (means + 0.3).requires_grad_()
        new_log_std = log_std + 0.2
        evaluated = self._evaluate(moved, new_log_std, pres)
        pre = torch.stack(pres)
        std = torch.exp(new_log_std)
        rho = 0.85
        t = 10
        loc = moved[t, 0] + rho * (pre[t - 1, 0] - moved[t - 1, 0])
        speed = torch.distributions.Normal(loc, std[0] * math.sqrt(1 - rho * rho))
        rest = torch.distributions.Normal(moved[t, 1:], std[1:])
        base = speed.log_prob(pre[t, 0]) + rest.log_prob(pre[t, 1:]).sum()
        correction = 2.0 * (math.log(2.0) - pre[t] - torch.nn.functional.softplus(-2.0 * pre[t]))
        expected = base - correction.sum()
        assert evaluated[t].item() == pytest.approx(expected.item(), abs=1e-5)
        (grad,) = torch.autograd.grad(evaluated[t], moved)
        # The previous step's mean enters with -rho times the speed's score.
        score = (pre[t, 0] - loc) / (std[0] ** 2 * (1 - rho * rho))
        assert grad[t - 1, 0].item() == pytest.approx((-rho * score).item(), rel=1e-4)
        assert grad[t, 0].item() == pytest.approx(score.item(), rel=1e-4)


class TestTheBuffer:
    def test_the_previous_records_follow_the_shuffle(self) -> None:
        buffer = RolloutBuffer(
            12,
            torch.device("cpu"),
            np.random.default_rng(0),
            continuous_actions=True,
        )
        for i in range(12):
            state = np.full(4, float(i), dtype=np.float32)
            buffer.add(
                state,
                np.zeros(5, dtype=np.float32),
                torch.tensor(0.0),
                torch.tensor(0.0),
                0.0,
                False,  # noqa: FBT003
                previous=PreviousStep(i + 0.5, state * 2.0, live=i % 3 != 0),
            )
        returns, advantages = buffer.compute_returns_and_advantages(torch.tensor(0.0), 0.99, 0.95)
        for minibatch in buffer.get_minibatches(3, returns, advantages):
            index = minibatch["states"][:, 0]
            assert torch.equal(minibatch["previous_speeds"], index + 0.5)
            assert torch.equal(minibatch["previous_states"][:, 0], index * 2.0)
            assert torch.equal(minibatch["previous_live"], index.long() % 3 != 0)

    def test_without_records_the_minibatches_are_unchanged(self) -> None:
        keys = {"states", "actions", "old_log_probs", "returns", "advantages"}
        buffer = RolloutBuffer(
            6,
            torch.device("cpu"),
            np.random.default_rng(0),
            continuous_actions=True,
        )
        for i in range(6):
            buffer.add(
                np.full(4, float(i), dtype=np.float32),
                np.zeros(5, dtype=np.float32),
                torch.tensor(0.0),
                torch.tensor(0.0),
                0.0,
                False,  # noqa: FBT003
            )
        returns, advantages = buffer.compute_returns_and_advantages(torch.tensor(0.0), 0.99, 0.95)
        assert all(set(mb) == keys for mb in buffer.get_minibatches(2, returns, advantages))


class TestTheEncoder:
    """Scenarios: Off is byte-identical; On appends one parameter."""

    def test_on_appends_one_parameter(self) -> None:
        off, on = _brain(), _brain(**_SLOW)
        assert len(off.topology.encoder.entries) == 196  # type: ignore[union-attr]
        assert len(on.topology.encoder.entries) == 197  # type: ignore[union-attr]
        assert on.topology.encoder.entries[-1].name == enc.SPEED_NOISE_ENTRY  # type: ignore[union-attr]
        tied = _brain(encoder_modulator_tying="per_ligand", **_SLOW)
        assert len(tied.topology.encoder.entries) == 133  # type: ignore[union-attr]
        assert off._speed_noise_rho is None
        assert "speed_noise_timescale_s" not in off.topology.state_dict()

    def test_a_196_vector_is_a_prefix_writing_the_same_tensors(self) -> None:
        off, on = _brain(), _brain(**_SLOW)
        z = _vector(off)
        enc.write_encoder(off, z)
        enc.write_encoder(on, [*z, 0.3])
        on_state, off_state = on.topology.state_dict(), off.topology.state_dict()
        assert set(on_state) - set(off_state) == {"encoder.speed_noise_timescale_s"}
        for name, tensor in off_state.items():
            assert torch.equal(tensor, on_state[name]), name
        assert np.allclose(enc.read_encoder(on)[:-1], enc.read_encoder(off))

    def test_the_centre_and_a_round_trip(self) -> None:
        on = _brain(**_SLOW)
        tau = math.sqrt(0.25 * 300.0)
        assert float(on.topology.encoder.speed_noise_timescale_s) == pytest.approx(tau, rel=1e-6)  # type: ignore[union-attr]
        assert on._speed_noise_rho == pytest.approx(math.exp(-1.0 / tau), rel=1e-6)
        z = _vector(on)
        enc.write_encoder(on, z)
        assert np.allclose(enc.read_encoder(on), z, atol=1e-6)


class TestUnderPPO:
    """Scenarios: tau is held under PPO; PPO evaluates the exact conditional."""

    def test_tau_is_held_and_an_update_runs(self) -> None:
        brain = _brain(
            freeze_updates=False,
            rollout_buffer_size=8,
            num_minibatches=2,
            num_epochs=1,
            **_SLOW,
        )
        enc.write_encoder(brain, _vector(brain))
        learnable = brain.topology.learnable_parameters
        assert sum(p.numel() for p in learnable) == 197 - 3
        tau = brain.topology.encoder.speed_noise_timescale_s.clone()  # type: ignore[union-attr]
        before = enc.read_encoder(brain)
        brain.prepare_episode()
        for t in range(9):
            brain.run_brain(_params(t), top_only=False, top_randomize=False)
            brain.learn(_params(t), reward=0.1 * (t % 3) - 0.1)
        assert not np.allclose(before, enc.read_encoder(brain))
        assert torch.equal(brain.topology.encoder.speed_noise_timescale_s, tau)  # type: ignore[union-attr]
        assert all(math.isfinite(loss) for loss in brain.history_data.losses)

    def test_before_an_update_every_ratio_is_one(self) -> None:
        from quantumnematode.learning_rules.ppo import ConnectomePPOBatch

        brain = _brain(freeze_updates=False, rollout_buffer_size=64, **_SLOW)
        enc.write_encoder(brain, _vector(brain))
        brain.prepare_episode()
        for t in range(20):
            brain.run_brain(_params(t), top_only=False, top_randomize=False)
            brain.learn(_params(t), reward=0.0)
        buffer = brain.buffer
        assert sum(buffer.previous_live) == 19
        batch = ConnectomePPOBatch(
            buffer=buffer,
            unpack_batched=brain._unpack_state_batched,
            unpack_modulators=brain._unpack_modulators,
            last_value=torch.tensor(0.0),
            speed_noise_rho=brain._speed_noise_rho,
        )
        returns, advantages = buffer.compute_returns_and_advantages(torch.tensor(0.0), 0.99, 0.95)
        rule = brain._ppo_rule
        assert rule is not None
        for minibatch in buffer.get_minibatches(1, returns, advantages):
            evaluated = rule._evaluate(brain.topology, batch, minibatch, [])
            assert torch.allclose(evaluated.new_log_probs, minibatch["old_log_probs"], atol=1e-4)


class TestRefusals:
    """Scenario: Unsupported settings are refused."""

    def test_slow_noise_without_the_encoder_is_refused(self) -> None:
        with pytest.raises(ValueError, match="read only with encoder='compact'"):
            ConnectomePPOBrain(config=_config(encoder="none", **_SLOW), device=DeviceType.CPU)

    def test_a_weight_file_from_the_other_mode_is_refused(self) -> None:
        on, off = _brain(**_SLOW), _brain()
        with pytest.raises(ValueError, match="encoder_speed_noise"):
            off.load_weight_components(on.get_weight_components())
        with pytest.raises(ValueError, match="encoder_speed_noise"):
            on.load_weight_components(off.get_weight_components())
        on.load_weight_components(_brain(**_SLOW).get_weight_components())
