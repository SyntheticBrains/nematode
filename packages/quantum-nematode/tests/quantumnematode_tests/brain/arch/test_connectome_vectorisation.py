"""Equivalence tests for the vectorised ConnectomePPO forward pass.

The PPO update was vectorised — `forward_with_hidden_batched` replaces the
per-sample Python loop over `forward_with_hidden`. Batched matmul reorders
float accumulation vs B separate matvecs, so the two paths are NOT bit-identical
but ARE mathematically equivalent. These tests assert the batched path matches
the single-sample path within float32 tolerance, across all four projection
configurations + every ContactZone, so a real wiring bug (not just float
reordering) is caught far tighter than any end-to-end training-success check.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import torch
import torch.nn.functional as f
from quantumnematode.brain.arch.connectome_ppo import (
    _N_CONTACT_ZONES,
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.env.env import ContactZone

_SEED = 2026
_ZONES = (ContactZone.NONE, ContactZone.ANTERIOR, ContactZone.POSTERIOR, ContactZone.LATERAL)
# float32 accumulation over 302 neurons across K recurrence steps: ~1e-6 drift expected.
_ATOL_LOGITS = 1e-5
_ATOL_HIDDEN = 1e-4


def _make_brain(**flags: object) -> ConnectomePPOBrain:
    cfg = ConnectomePPOBrainConfig(
        seed=_SEED,
        sensing_mode="klinotaxis",
        forward_pass_depth=4,
        **flags,  # type: ignore[arg-type]
    )
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _batched_inputs(
    brain: ConnectomePPOBrain,
    batch: int,
    *,
    zone_idx: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    cfg = brain.config
    torch.manual_seed(0)
    food = torch.randn(batch, brain._n_food_features)
    distal = mechano = zone_oh = thermo = None
    if cfg.enable_predator_projection:
        distal = torch.randn(batch, 2)
        mechano = torch.randn(batch, 2)
        idx = zone_idx if zone_idx is not None else torch.randint(0, _N_CONTACT_ZONES, (batch,))
        zone_oh = f.one_hot(idx, _N_CONTACT_ZONES).float()
    if cfg.enable_thermotaxis_projection:
        thermo = torch.randn(batch, 3)
    return {
        "food": food,
        "distal": distal,
        "mechano": mechano,
        "zone_oh": zone_oh,
        "thermo": thermo,
    }


def _single_sample_loop(
    brain: ConnectomePPOBrain,
    inp: dict[str, torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    topo = brain.topology
    food = inp["food"]
    assert food is not None
    batch = food.shape[0]
    logits = torch.empty(batch, 4)
    hidden = torch.empty(batch, topo.n_neurons)
    for k in range(batch):
        zone_oh = inp["zone_oh"]
        zone = _ZONES[int(zone_oh[k].argmax())] if zone_oh is not None else None
        lk, hk = topo.forward_with_hidden(
            food[k],
            predator_distal_features=None if inp["distal"] is None else inp["distal"][k],
            predator_mechano_features=None if inp["mechano"] is None else inp["mechano"][k],
            predator_contact_zone=zone,
            thermotaxis_features=None if inp["thermo"] is None else inp["thermo"][k],
        )
        logits[k] = lk
        hidden[k] = hk
    return logits, hidden


def _assert_batched_matches_single(
    brain: ConnectomePPOBrain,
    inp: dict[str, torch.Tensor | None],
) -> None:
    food = inp["food"]
    assert food is not None
    bl, bh = brain.topology.forward_with_hidden_batched(
        food,
        predator_distal_features=inp["distal"],
        predator_mechano_features=inp["mechano"],
        contact_zone_onehot=inp["zone_oh"],
        thermotaxis_features=inp["thermo"],
    )
    sl, sh = _single_sample_loop(brain, inp)
    assert torch.allclose(bl, sl, atol=_ATOL_LOGITS), (
        f"batched logits diverge from single-sample by {(bl - sl).abs().max().item():.2e}"
    )
    assert torch.allclose(bh, sh, atol=_ATOL_HIDDEN), (
        f"batched hidden diverges from single-sample by {(bh - sh).abs().max().item():.2e}"
    )


class TestBatchedForwardMatchesSingleSample:
    """The batched forward is numerically equivalent to the per-sample loop."""

    def test_foraging_only(self) -> None:
        brain = _make_brain()
        _assert_batched_matches_single(brain, _batched_inputs(brain, 16))

    def test_predator_only(self) -> None:
        brain = _make_brain(enable_predator_projection=True)
        _assert_batched_matches_single(brain, _batched_inputs(brain, 16))

    def test_thermo_only(self) -> None:
        brain = _make_brain(enable_thermotaxis_projection=True)
        _assert_batched_matches_single(brain, _batched_inputs(brain, 16))

    def test_combined(self) -> None:
        brain = _make_brain(enable_predator_projection=True, enable_thermotaxis_projection=True)
        _assert_batched_matches_single(brain, _batched_inputs(brain, 16))

    def test_each_contact_zone_routes_identically(self) -> None:
        """Every ContactZone's masked-batched injection matches the per-zone branch."""
        brain = _make_brain(enable_predator_projection=True, enable_thermotaxis_projection=True)
        for z in range(_N_CONTACT_ZONES):
            zone_idx = torch.full((12,), z, dtype=torch.long)
            _assert_batched_matches_single(brain, _batched_inputs(brain, 12, zone_idx=zone_idx))


class TestPoolMotorHandlesBatchAndSingle:
    """The unified _pool_motor pools over the last dim for 1D and 2D inputs."""

    def test_single_and_batched_agree(self) -> None:
        brain = _make_brain()
        topo = brain.topology
        torch.manual_seed(1)
        h_batch = torch.randn(8, topo.n_neurons)
        pooled_batch = topo._pool_motor(h_batch)
        assert pooled_batch.shape == (8, 4)
        for k in range(8):
            pooled_single = topo._pool_motor(h_batch[k])
            assert pooled_single.shape == (4,)
            assert torch.equal(pooled_single, pooled_batch[k])


class TestUnpackStateBatchedRoundTrip:
    """_unpack_state_batched slices the same layout as the single-sample _unpack_state."""

    def test_combined_layout(self) -> None:
        brain = _make_brain(enable_predator_projection=True, enable_thermotaxis_projection=True)
        torch.manual_seed(2)
        # [food(3) + predator(8) + thermo(3)] = 14
        states = torch.randn(5, 14)
        # zone one-hot must be a valid one-hot for the argmax in the single path.
        states[:, 7:11] = f.one_hot(torch.tensor([0, 1, 2, 3, 1]), _N_CONTACT_ZONES).float()
        food_b, distal_b, mechano_b, zone_oh_b, thermo_b = brain._unpack_state_batched(states)
        assert distal_b is not None
        assert mechano_b is not None
        assert zone_oh_b is not None
        assert thermo_b is not None
        assert food_b.shape == (5, 3)
        assert distal_b.shape == (5, 2)
        assert mechano_b.shape == (5, 2)
        assert zone_oh_b.shape == (5, _N_CONTACT_ZONES)
        assert thermo_b.shape == (5, 3)
        # Row-wise agreement with the single-sample unpack.
        for k in range(5):
            food_s, distal_s, mechano_s, _zone_s, thermo_s = brain._unpack_state(states[k])
            assert distal_s is not None
            assert mechano_s is not None
            assert thermo_s is not None
            assert torch.equal(food_b[k], food_s)
            assert torch.equal(distal_b[k], distal_s)
            assert torch.equal(mechano_b[k], mechano_s)
            assert torch.equal(thermo_b[k], thermo_s)
