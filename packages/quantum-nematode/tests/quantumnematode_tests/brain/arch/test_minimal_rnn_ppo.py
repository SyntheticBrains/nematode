"""Tests for the minimal-RNN (minGRU / minLSTM) PPO arms.

Covers the cell recurrence (shapes, input-only gating, bounded single state), config
validation (rejecting the unhonoured plain-RNN cell fields), the registry round-trip, a
forward/learn smoke in both action modes, and a weight-persistence round-trip.
"""

from __future__ import annotations

import pytest
import torch
from quantumnematode.brain.arch import (
    MinGRUPPOBrain,
    MinGRUPPOBrainConfig,
    MinLSTMPPOBrain,
    MinLSTMPPOBrainConfig,
)
from quantumnematode.brain.arch._brain import BrainParams
from quantumnematode.brain.arch._registry import instantiate_brain
from quantumnematode.brain.arch.dtypes import BrainType
from quantumnematode.brain.arch.minimal_rnn_ppo import MinimalRNN
from quantumnematode.brain.modules import ModuleName
from quantumnematode.utils.config_loader import BRAIN_CONFIG_MAP

_MODS = [ModuleName.CUE, ModuleName.GO_SIGNAL]


def _cfg(cls, **kw):
    """Return a small valid config for the given minimal-RNN config class."""
    return cls(
        sensory_modules=_MODS,
        lstm_hidden_dim=8,
        bptt_chunk_length=4,
        rollout_buffer_size=8,
        **kw,
    )


# ── Cell ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("is_lstm", [False, True])
def test_cell_output_shapes(is_lstm):
    """forward(x_seq, h) returns (seq,1,hidden) output and a (1,1,hidden) single state."""
    cell = MinimalRNN(input_size=3, hidden_size=5, is_lstm=is_lstm)
    x_seq = torch.randn(7, 1, 3)
    h0 = torch.zeros(1, 1, 5)
    output, h_new = cell(x_seq, h0)
    assert output.shape == (7, 1, 5)
    assert h_new.shape == (1, 1, 5)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("is_lstm", [False, True])
def test_cell_gates_are_input_only(is_lstm):
    """The h0 -> h_T map is affine (gates depend on input only, not the hidden state).

    With identical inputs, equally-spaced initial states must map to equally-spaced final
    states: h_T(2H) - h_T(H) == h_T(H) - h_T(0). This holds iff the per-step coefficient on
    the previous state depends only on the input — the defining minimal-RNN property.
    """
    torch.manual_seed(0)
    cell = MinimalRNN(input_size=3, hidden_size=5, is_lstm=is_lstm)
    x_seq = torch.randn(6, 1, 3)
    base = torch.randn(1, 1, 5)

    def final(h0: torch.Tensor) -> torch.Tensor:
        return cell(x_seq, h0)[1]

    h_a = final(torch.zeros(1, 1, 5))
    h_b = final(base)
    h_c = final(2 * base)
    torch.testing.assert_close(h_c - h_b, h_b - h_a, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("is_lstm", [False, True])
def test_cell_state_is_bounded_by_candidates(is_lstm):
    """From h0=0 the state is a convex combination of candidates, so |h| <= max|h_tilde|.

    A long input sequence must not let the (un-squashed) state blow up — the convex update is
    what keeps it bounded without a nonlinearity.
    """
    torch.manual_seed(1)
    cell = MinimalRNN(input_size=4, hidden_size=6, is_lstm=is_lstm)
    x_seq = torch.randn(400, 1, 4)
    _, h_final = cell(x_seq, torch.zeros(1, 1, 6))
    # Candidate magnitudes across the run bound the convex-combined state.
    candidates = cell.weight_h(x_seq)  # (seq, 1, hidden)
    max_candidate = candidates.abs().max().item()
    assert torch.isfinite(h_final).all()
    assert h_final.abs().max().item() <= max_candidate + 1e-5


def test_minlstm_normalization_bounds_state_with_high_gates():
    """The minLSTM `f/(f+i)` normalization is what keeps the state bounded.

    With both gates driven high (`bias_f = bias_i` large), an UN-normalized update
    `h = f*h_prev + i*h_tilde` has `f + i ≈ 2`, so the state grows ~2x per step and diverges. The
    shipped normalized update (`f' + i' ≈ 1`) stays within the candidate range. This exercises the
    regime the random-input bounded test never does — so a dropped normalization is caught here.
    """
    torch.manual_seed(2)
    cell = MinimalRNN(input_size=4, hidden_size=6, is_lstm=True)
    with torch.no_grad():
        cell.weight_f.bias.fill_(5.0)  # f ~ 0.99
        cell.weight_i.bias.fill_(5.0)  # i ~ 0.99 -> an un-normalized f+i ~ 2 would diverge
    x_seq = torch.randn(300, 1, 4)
    _, h_final = cell(x_seq, torch.zeros(1, 1, 6))
    max_candidate = cell.weight_h(x_seq).abs().max().item()
    assert torch.isfinite(h_final).all()
    assert h_final.abs().max().item() <= max_candidate + 1e-5


@pytest.mark.parametrize(
    ("brain_cls", "cfg_cls"),
    [(MinGRUPPOBrain, MinGRUPPOBrainConfig), (MinLSTMPPOBrain, MinLSTMPPOBrainConfig)],
)
def test_retention_gate_defaults_to_hold(brain_cls, cfg_cls):
    """The retention gate is biased toward HOLDING at init (a memory-friendly prior).

    During a zero-input phase the gate is bias-only; a zeroed bias gives a ~1-step retention
    half-life that washes out a held signal. The hold-bias init retains most of the prior state
    across a zero-input step, so the policy only has to learn to write during the input.
    """
    brain = brain_cls(_cfg(cfg_cls), num_actions=4)
    rnn = brain.rnn
    assert isinstance(rnn, MinimalRNN)
    h_prev = torch.ones(1, 1, brain.config.lstm_hidden_dim)
    x0 = torch.zeros(1, 1, brain.input_dim)
    _, h_new = rnn(x0, h_prev)
    # With zero input the candidate is the (zeroed) bias, so h_new = retention_coeff * h_prev:
    # a high fraction of the all-ones prior (~0.92), never grown beyond it. The two bounds catch
    # every regression mode: a zeroed update-gate bias -> 0.5 (< 0.8); a retain<->write direction
    # swap -> ~0.08 (< 0.8); and a non-holding cell whose candidate bias inflates the output -> > 1.
    assert ((h_new > 0.8) & (h_new <= 1.0 + 1e-5)).all()


# ── Config validation ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [MinGRUPPOBrainConfig, MinLSTMPPOBrainConfig])
def test_config_requires_sensory_modules(cls):
    """sensory_modules is required (inherited from the LSTM-PPO config)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        cls(lstm_hidden_dim=8)


@pytest.mark.parametrize("cls", [MinGRUPPOBrainConfig, MinLSTMPPOBrainConfig])
@pytest.mark.parametrize("field", ["rnn_type", "recurrent_layernorm"])
def test_config_rejects_unhonoured_plain_rnn_fields(cls, field):
    """A non-default rnn_type / recurrent_layernorm fails loudly (they are not honoured)."""
    from pydantic import ValidationError

    value = "gru" if field == "rnn_type" else True  # non-default values
    with pytest.raises(ValidationError):
        cls(sensory_modules=_MODS, **{field: value})


@pytest.mark.parametrize("cls", [MinGRUPPOBrainConfig, MinLSTMPPOBrainConfig])
def test_config_defaults_are_accepted(cls):
    """Not setting the plain-RNN fields is fine (their ignored defaults pass validation)."""
    cfg = _cfg(cls)
    assert cfg.sensory_modules == _MODS


# ── Registry round-trip ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "brain_type", "config_cls", "brain_cls"),
    [
        ("mingruppo", BrainType.MINGRU_PPO, MinGRUPPOBrainConfig, MinGRUPPOBrain),
        ("minlstmppo", BrainType.MINLSTM_PPO, MinLSTMPPOBrainConfig, MinLSTMPPOBrain),
    ],
)
def test_registry_round_trip(name, brain_type, config_cls, brain_cls):
    """Name <-> BrainType <-> config are consistent and instantiate the right brain class."""
    assert brain_type.value == name
    assert BRAIN_CONFIG_MAP[name] is config_cls
    # The registry instantiates the registered class for this name.
    assert isinstance(instantiate_brain(name, _cfg(config_cls), num_actions=4), brain_cls)
    # Single-state invariants, checked on a concretely-typed instance.
    brain = brain_cls(_cfg(config_cls), num_actions=4)
    assert brain._is_gru is True  # single recurrent state
    assert brain.c_t is None
    assert isinstance(brain.rnn, MinimalRNN)


# ── Forward / learn smoke ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("brain_cls", [MinGRUPPOBrain, MinLSTMPPOBrain])
def test_discrete_forward_learn(brain_cls):
    """One discrete run_brain -> learn cycle emits an action and accepts a reward."""
    cfg = _cfg(MinGRUPPOBrainConfig if brain_cls is MinGRUPPOBrain else MinLSTMPPOBrainConfig)
    brain = brain_cls(cfg, num_actions=4)
    params = BrainParams(cue_signal=1.0, go_signal=0.0)
    out = brain.run_brain(
        params=params,
        reward=0.0,
        input_data=None,
        top_only=True,
        top_randomize=True,
    )
    assert len(out) == 1
    assert out[0].action is not None
    brain.learn(params=params, reward=1.0, episode_done=False)


@pytest.mark.parametrize("brain_cls", [MinGRUPPOBrain, MinLSTMPPOBrain])
def test_continuous_forward_learn(brain_cls):
    """One continuous run_brain -> learn cycle emits a (speed, turn) action."""
    cfg_cls = MinGRUPPOBrainConfig if brain_cls is MinGRUPPOBrain else MinLSTMPPOBrainConfig
    brain = brain_cls(_cfg(cfg_cls, action_mode="continuous"), num_actions=4)
    params = BrainParams(cue_signal=-1.0, go_signal=1.0)
    out = brain.run_brain(
        params=params,
        reward=0.0,
        input_data=None,
        top_only=True,
        top_randomize=True,
    )
    assert len(out) == 1
    assert out[0].continuous is not None
    assert len(out[0].continuous) == 2
    brain.learn(params=params, reward=1.0, episode_done=False)


# ── Weight persistence ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("brain_cls", "cfg_cls"),
    [(MinGRUPPOBrain, MinGRUPPOBrainConfig), (MinLSTMPPOBrain, MinLSTMPPOBrainConfig)],
)
def test_weight_persistence_round_trip(brain_cls, cfg_cls):
    """get/load_weight_components restores the minimal core (under the inherited 'lstm' key)."""
    src = brain_cls(_cfg(cfg_cls), num_actions=4)
    components = src.get_weight_components()
    assert "lstm" in components  # the recurrent core serializes under the inherited key

    dst = brain_cls(_cfg(cfg_cls), num_actions=4)
    # Pre-condition: fresh init differs from the source on at least one core weight.
    assert not torch.allclose(src.rnn.weight_h.weight, dst.rnn.weight_h.weight)

    dst.load_weight_components(components)
    # Every recurrent-core param round-trips — the gate projections AND all biases (incl. the
    # load-bearing hold-bias), not just one weight.
    src_state = src.rnn.state_dict()
    dst_state = dst.rnn.state_dict()
    assert set(src_state) == set(dst_state)
    for key, src_param in src_state.items():
        torch.testing.assert_close(dst_state[key], src_param)
