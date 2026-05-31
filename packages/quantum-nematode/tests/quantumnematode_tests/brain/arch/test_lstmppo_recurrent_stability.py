"""Tests for the LSTMPPO recurrent-stability fixes.

Two changes are covered:

- **Orthogonal recurrent init** — the GRU/LSTM hidden-to-hidden weights are
  orthogonally initialised per gate block (PyTorch leaves them at uniform init,
  which causes recurrent-state saturation that collapses the policy on a large
  fraction of seeds).
- **LayerNorm recurrent cell** — an optional ``recurrent_layernorm`` config flag
  swaps ``nn.GRU``/``nn.LSTM`` for a custom cell with LayerNorm on the gate
  pre-activations (Ba et al. 2016). Default off → byte-identical to before.
"""

from __future__ import annotations

import torch
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.lstmppo import (
    LayerNormRecurrent,
    LSTMPPOBrain,
    LSTMPPOBrainConfig,
)
from quantumnematode.brain.modules import ModuleName
from torch import nn

_SENSORY = [ModuleName.FOOD_CHEMOTAXIS, ModuleName.PROPRIOCEPTION]
_HIDDEN = 16
_ATOL = 1e-5


def _make_brain(**overrides: object) -> LSTMPPOBrain:
    cfg = LSTMPPOBrainConfig(
        sensory_modules=_SENSORY,
        lstm_hidden_dim=_HIDDEN,
        seed=42,
        **overrides,  # type: ignore[arg-type]
    )
    return LSTMPPOBrain(config=cfg, num_actions=4, device=DeviceType.CPU)


class TestOrthogonalRecurrentInit:
    """The recurrent hidden-to-hidden weights are orthogonal per gate block."""

    def _assert_hh_orthogonal(self, brain: LSTMPPOBrain, n_gates: int) -> None:
        hh = dict(brain.rnn.named_parameters())[
            "weight_hh_l0" if hasattr(brain.rnn, "weight_hh_l0") else "weight_hh"
        ]
        for g in range(n_gates):
            block = hh.data[g * _HIDDEN : (g + 1) * _HIDDEN]  # (hidden, hidden)
            prod = block @ block.t()
            assert torch.allclose(prod, torch.eye(_HIDDEN), atol=_ATOL), (
                f"gate block {g} not orthogonal (max dev "
                f"{(prod - torch.eye(_HIDDEN)).abs().max().item():.2e})"
            )

    def test_gru_recurrent_weights_orthogonal(self) -> None:
        """GRU weight_hh (3 gate blocks) SHALL each be orthogonal after init."""
        self._assert_hh_orthogonal(_make_brain(rnn_type="gru"), n_gates=3)

    def test_lstm_recurrent_weights_orthogonal(self) -> None:
        """LSTM weight_hh (4 gate blocks) SHALL each be orthogonal after init."""
        self._assert_hh_orthogonal(_make_brain(rnn_type="lstm"), n_gates=4)


class TestLayerNormFlag:
    """The ``recurrent_layernorm`` flag swaps the recurrent cell, default off."""

    def test_default_uses_pytorch_rnn(self) -> None:
        """recurrent_layernorm defaults False → self.rnn is nn.GRU / nn.LSTM."""
        assert isinstance(_make_brain(rnn_type="gru").rnn, nn.GRU)
        assert isinstance(_make_brain(rnn_type="lstm").rnn, nn.LSTM)

    def test_flag_on_uses_layernorm_cell(self) -> None:
        """recurrent_layernorm=True → self.rnn is LayerNormRecurrent (both cell types)."""
        for rt in ("gru", "lstm"):
            brain = _make_brain(rnn_type=rt, recurrent_layernorm=True)
            assert isinstance(brain.rnn, LayerNormRecurrent)
            assert brain.rnn.is_gru == (rt == "gru")


class TestLayerNormRecurrentForward:
    """The LayerNormRecurrent forward matches the nn.GRU/nn.LSTM interface."""

    def test_gru_forward_shapes(self) -> None:
        """GRU mode: (x_seq, h) → (output (seq,batch,hidden), h_new (1,batch,hidden))."""
        cell = LayerNormRecurrent(8, _HIDDEN, is_gru=True)
        x_seq = torch.randn(5, 2, 8)  # (seq_len, batch, input)
        h = torch.zeros(1, 2, _HIDDEN)
        out, h_new = cell(x_seq, h)
        assert out.shape == (5, 2, _HIDDEN)
        assert h_new.shape == (1, 2, _HIDDEN)
        assert torch.isfinite(out).all()
        # h_new is the last output step (GRU returns the final hidden as output[-1]).
        assert torch.allclose(h_new.squeeze(0), out[-1], atol=_ATOL)

    def test_lstm_forward_shapes(self) -> None:
        """LSTM mode: (x_seq, (h,c)) → (output, (h_new (1,b,h), c_new (1,b,h)))."""
        cell = LayerNormRecurrent(8, _HIDDEN, is_gru=False)
        x_seq = torch.randn(5, 2, 8)
        h = torch.zeros(1, 2, _HIDDEN)
        c = torch.zeros(1, 2, _HIDDEN)
        out, (h_new, c_new) = cell(x_seq, (h, c))
        assert out.shape == (5, 2, _HIDDEN)
        assert h_new.shape == (1, 2, _HIDDEN)
        assert c_new.shape == (1, 2, _HIDDEN)
        assert torch.isfinite(out).all()
        assert torch.isfinite(c_new).all()
