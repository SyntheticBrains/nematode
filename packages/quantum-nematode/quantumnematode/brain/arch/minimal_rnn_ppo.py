"""Minimal-RNN (minGRU / minLSTM) PPO brain arms.

Two classical PPO arms backed by parallel-form minimal RNNs (Feng et al. 2024,
"Were RNNs All We Needed?", arXiv:2410.01201). The gates depend on the current input
**only** (no hidden-to-hidden matrix), so the recurrence is a convex state update — an
associative scan that stays bounded without a squashing nonlinearity — and both variants
carry a single recurrent state.

The arms subclass :class:`LSTMPPOBrain` and override only the recurrent-core construction
and its weight init; the whole PPO / chunk-BPTT / rollout-buffer / weight-persistence
pipeline is reused unchanged. The absence of a saturating hidden-to-hidden matrix is also
the principled reason these arms are a candidate stability upgrade to the plain LSTM arm.
"""

from __future__ import annotations

import torch
from pydantic import model_validator
from torch import nn

from quantumnematode.brain.arch._registry import register_brain
from quantumnematode.brain.arch.dtypes import BrainType
from quantumnematode.brain.arch.lstmppo import LSTMPPOBrain, LSTMPPOBrainConfig

# Floor for the minLSTM gate normalisation f/(f+i). With sigmoid gates f+i is in (0, 2),
# so this only guards the degenerate f,i -> 0 limit; it never meaningfully shifts the gates.
_GATE_NORM_EPS = 1e-8


class MinimalRNN(nn.Module):
    """Single-state, input-only-gate minimal RNN cell (minGRU / minLSTM).

    A drop-in for the GRU branch used by :class:`LSTMPPOBrain`:
    ``forward(x_seq, h) -> (output_seq, h_new)`` with ``x_seq`` shaped
    ``(seq_len, batch, input_size)`` and ``h`` shaped ``(1, batch, hidden_size)``.

    Recurrences (``x`` = the current input; all gates are functions of ``x`` only):

    - **minGRU:** ``z = sigmoid(W_z x)``, ``h_tilde = W_h x``,
      ``h = (1 - z) * h_prev + z * h_tilde``.
    - **minLSTM:** ``f = sigmoid(W_f x)``, ``i = sigmoid(W_i x)``, normalised
      ``f' = f/(f+i)``, ``i' = i/(f+i)``, ``h_tilde = W_h x``,
      ``h = f' * h_prev + i' * h_tilde``.

    The minGRU update is exactly convex (``(1 - z) + z = 1``); the minLSTM update is contractive
    (the normalised coefficients sum to ``(f + i) / (f + i + eps) ≤ 1``). Either way the state
    stays within the range of ``h_tilde`` — affine in the bounded LayerNorm'd input — and remains
    bounded over arbitrary sequence lengths without a squashing nonlinearity. There is no
    hidden-to-hidden (``weight_hh``) matrix, which is why the arms override
    ``_init_recurrent_weights`` with an input-projection-only initialisation.
    """

    def __init__(self, input_size: int, hidden_size: int, *, is_lstm: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.is_lstm = is_lstm
        self.weight_h = nn.Linear(input_size, hidden_size)  # candidate projection (both variants)
        if is_lstm:
            self.weight_f = nn.Linear(input_size, hidden_size)
            self.weight_i = nn.Linear(input_size, hidden_size)
        else:
            self.weight_z = nn.Linear(input_size, hidden_size)

    def forward(
        self,
        x_seq: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll the cell over the sequence; returns ``(output_seq, h_new)`` like ``nn.GRU``.

        Parameters
        ----------
        x_seq : torch.Tensor
            Input sequence, shape ``(seq_len, batch, input_size)``.
        hidden : torch.Tensor
            Initial hidden state, shape ``(1, batch, hidden_size)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(output, h_new)`` where ``output`` is ``(seq_len, batch, hidden_size)`` and
            ``h_new`` is ``(1, batch, hidden_size)``.
        """
        # The gates and the candidate are input-only (no dependence on the hidden state), so
        # project the WHOLE sequence once — the parallel-form property — and keep only the
        # elementwise convex state update in the per-step loop. Both variants reduce to
        # ``h = retain * h_prev + write * h_tilde``.
        h_tilde = self.weight_h(x_seq)  # (seq, batch, hidden)
        if self.is_lstm:
            f = torch.sigmoid(self.weight_f(x_seq))
            i = torch.sigmoid(self.weight_i(x_seq))
            denom = f + i + _GATE_NORM_EPS
            retain, write = f / denom, i / denom
        else:
            write = torch.sigmoid(self.weight_z(x_seq))  # z
            retain = 1.0 - write
        h = hidden.squeeze(0)  # (batch, hidden)
        outputs = []
        for t in range(x_seq.shape[0]):
            h = retain[t] * h + write[t] * h_tilde[t]
            outputs.append(h)
        output = torch.stack(outputs, dim=0)  # (seq_len, batch, hidden)
        return output, h.unsqueeze(0)


class _MinimalRNNPPOConfig(LSTMPPOBrainConfig):
    """Shared configuration for the minimal-RNN arms (inherits the LSTM-PPO surface).

    The minimal arms always use the single-state minimal core, so the inherited
    ``rnn_type`` / ``recurrent_layernorm`` (plain-RNN cell-selection) fields are not
    honoured. Explicitly setting either is rejected so a stray value fails loudly rather
    than silently selecting a different, two-state path.
    """

    @model_validator(mode="after")
    def _reject_plain_rnn_cell_fields(self) -> _MinimalRNNPPOConfig:
        # rnn_type / recurrent_layernorm select the plain-RNN cell and are NOT honoured by the
        # minimal arms (which always use the single-state minimal core). The config loader
        # repopulates every field from its default, so ``model_fields_set`` is unreliable here —
        # reject a non-default *value* instead (the only way a user signals real intent).
        if self.rnn_type != "lstm" or self.recurrent_layernorm:
            msg = (
                f"{type(self).__name__} does not honour rnn_type / recurrent_layernorm — the "
                "minimal-RNN arms always use the single-state minimal core. Remove these "
                f"fields (got rnn_type={self.rnn_type!r}, "
                f"recurrent_layernorm={self.recurrent_layernorm})."
            )
            raise ValueError(msg)
        return self


class MinGRUPPOBrainConfig(_MinimalRNNPPOConfig):
    """Configuration for the minGRU PPO arm (``name: mingruppo``)."""


class MinLSTMPPOBrainConfig(_MinimalRNNPPOConfig):
    """Configuration for the minLSTM PPO arm (``name: minlstmppo``)."""


class _MinimalRNNPPOBrain(LSTMPPOBrain):
    """Shared base for the minimal-RNN arms: build the minimal core + init its projections.

    Overrides only the two recurrent-core hooks exposed by :class:`LSTMPPOBrain`; every
    other part of the PPO / chunk-BPTT / buffer / persistence pipeline is inherited. The
    single recurrent state means the inherited GRU-shaped path (``c = None``) is reused, so
    ``self._is_gru`` is pinned ``True`` regardless of the (ignored) inherited ``rnn_type``.
    """

    _MINIMAL_IS_LSTM: bool = False
    _MINIMAL_LABEL: str = "minrnn"

    def _build_recurrent_core(self) -> nn.Module:
        """Build the minimal-RNN core and pin the single-state path + log label."""
        self._is_gru = True  # single recurrent state -> reuse the GRU-shaped (c=None) path
        self._recurrent_core_label = self._MINIMAL_LABEL
        return MinimalRNN(
            self.input_dim,
            self.config.lstm_hidden_dim,
            is_lstm=self._MINIMAL_IS_LSTM,
        ).to(self.device)

    # Default-to-hold gate bias. During a zero-input phase (e.g. the bit-memory delay, obs
    # [0, 0]) the gate is bias-only, so a zeroed bias gives z = f' = 0.5 — a ~1-step retention
    # half-life that washes out any held signal. Biasing the retention gate toward holding (the
    # LSTM forget-gate-bias trick adapted to the minimal cell) extends the half-life so the
    # policy only has to learn to WRITE during the input, not discover the hold via delayed
    # credit. z ~ sigmoid(-2.5) ~ 0.08 -> retain ~0.92; minLSTM f' ~ 0.92 likewise.
    _HOLD_BIAS = 2.5

    def _init_recurrent_weights(self) -> None:
        """Xavier-init the input projections; bias the retention gate toward holding.

        The minimal RNN has no hidden-to-hidden matrix, so the base orthogonal-``weight_hh``
        pass (which fights recurrent-state saturation in the plain GRU/LSTM) does not apply.
        The retention-gate bias (see ``_HOLD_BIAS``) gives the cell a memory-friendly prior.
        """
        for name, param in self.rnn.named_parameters():
            if "weight_z.bias" in name:  # minGRU update gate: small z -> large retention (1-z)
                nn.init.constant_(param.data, -self._HOLD_BIAS)
            elif "weight_f.bias" in name:  # minLSTM forget gate: large f -> large retention f'
                nn.init.constant_(param.data, self._HOLD_BIAS)
            elif "weight_i.bias" in name:  # minLSTM input gate: small i -> f' -> 1
                nn.init.constant_(param.data, -self._HOLD_BIAS)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)
            else:
                nn.init.xavier_uniform_(param.data)


@register_brain(
    name="mingruppo",
    config_cls=MinGRUPPOBrainConfig,
    brain_type=BrainType.MINGRU_PPO,
    families=("classical",),
)
class MinGRUPPOBrain(_MinimalRNNPPOBrain):
    """minGRU-augmented PPO arm (single update gate, single recurrent state)."""

    _MINIMAL_IS_LSTM = False
    _MINIMAL_LABEL = "mingru"


@register_brain(
    name="minlstmppo",
    config_cls=MinLSTMPPOBrainConfig,
    brain_type=BrainType.MINLSTM_PPO,
    families=("classical",),
)
class MinLSTMPPOBrain(_MinimalRNNPPOBrain):
    """minLSTM-augmented PPO arm (two normalised input-only gates, single recurrent state)."""

    _MINIMAL_IS_LSTM = True
    _MINIMAL_LABEL = "minlstm"
