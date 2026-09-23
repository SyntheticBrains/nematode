"""Chemical weight-draw structure, and what two wirings hold in common at one seed.

``weight_draw`` chooses the draw's *structure*, orthogonally to ``weight_init``'s scale rule.

Under the default ``edge_order`` the loop walks the ``(pre, post)``-sorted edge list drawing one
value per edge. Two graphs with the same edge count consume the same standard-normal stream, so the
nth *value* matches -- but the nth *edge* does not, so the value-to-edge pairing differs and so does
the per-edge scale sequence. That pairing is the residual confound in a wild-type-vs-rewired
contrast, and the two sharing modes remove it in the two ways that are defensible:

* ``dense_mask`` -- one dense draw read by ``(pre, post)``, so every edge present in **both** graphs
  carries the identical value.
* ``per_neuron_fanin`` -- one block per post-synaptic neuron assigned in pre-synaptic-index order,
  so every neuron receives the identical **multiset** of incoming weights and only which partner
  holds which value moves.

Neither is uniquely "the same initialisation" once the edge set changes, which is why both exist.

Every property below is **asserted against constructed brains**, never argued from how the stream is
consumed: a shared stream establishes only that the same values were drawn, not where they landed.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[6]
_ARMS = _REPO_ROOT / "configs" / "scenarios" / "foraging_predator_thermal"
_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis_plastic_"
_FROZEN = _ARMS / f"{_STEM}frozen.yml"
_FROZEN_REWIRED = _ARMS / f"{_STEM}frozen_rewired_null.yml"

_SEED = 23
_MODES = ("edge_order", "dense_mask", "per_neuron_fanin")
# The one tensor a draw mode is allowed to move. Everything else on the topology -- the readout,
# every sensor gain, the log-std -- comes from the torch global RNG, which this option never
# reaches. The tests below assert that over *every* remaining parameter rather than a hand-picked
# pair, so a tensor added later is covered without anyone remembering to add it here.
_DRAWN = "w_chem"


def _brain(path: Path, **overrides: object) -> ConnectomePPOBrain:
    container = load_simulation_config(str(path)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    cfg = container.config.model_copy(update={"seed": _SEED, **overrides})
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _chem(brain: ConnectomePPOBrain) -> torch.Tensor:
    return brain.topology.w_chem.detach()


def _mask(brain: ConnectomePPOBrain) -> torch.Tensor:
    return brain.topology.m_chem.detach().bool()


def _others(brain: ConnectomePPOBrain) -> dict[str, torch.Tensor]:
    """Every parameter the draw mode does not claim to touch, topology AND critic.

    The critic is constructed inside the learning rule, outside the topology, so a check confined to
    ``topology.named_parameters()`` would not see it move.
    """
    out = {
        f"topology.{name}": param.detach()
        for name, param in brain.topology.named_parameters()
        if name != _DRAWN
    }
    # The brain exposes the critic through a property that raises unless PPO is the learner, and
    # keeps the rule only when it is -- so the rule's presence is the guard. An earlier version
    # looked for a `rule` attribute the brain does not have, and so never compared the critic.
    if brain._ppo_rule is not None:
        out |= {f"critic.{n}": p.detach() for n, p in brain.critic.named_parameters()}
    return out


@pytest.fixture(scope="module")
def wild() -> dict[str, ConnectomePPOBrain]:
    """Wild-type brains, one per draw mode, all at the same seed."""
    return {mode: _brain(_FROZEN, weight_draw=mode) for mode in _MODES}


@pytest.fixture(scope="module")
def rewired() -> dict[str, ConnectomePPOBrain]:
    """Rewired-null brains, one per draw mode, all at the same seed."""
    return {mode: _brain(_FROZEN_REWIRED, weight_draw=mode) for mode in _MODES}


class TestDefaultIsUnchanged:
    """``edge_order`` is the pre-option brain, bit for bit."""

    def test_explicit_edge_order_is_bit_identical_to_the_default(self) -> None:
        default = _brain(_FROZEN)
        explicit = _brain(_FROZEN, weight_draw="edge_order")
        assert torch.equal(_chem(default), _chem(explicit))

    def test_every_parameter_is_bit_identical_to_the_default(self) -> None:
        default = dict(_brain(_FROZEN).topology.named_parameters())
        explicit = dict(_brain(_FROZEN, weight_draw="edge_order").topology.named_parameters())
        assert default.keys() == explicit.keys()
        for name, param in default.items():
            assert torch.equal(param, explicit[name]), f"{name} differs under an explicit default"


class TestDenseMaskSharesEveryCommonEdge:
    """Every edge present in both graphs carries the identical value."""

    def test_shared_edges_match_across_wirings(
        self,
        wild: dict[str, ConnectomePPOBrain],
        rewired: dict[str, ConnectomePPOBrain],
    ) -> None:
        wt, rn = wild["dense_mask"], rewired["dense_mask"]
        both = _mask(wt) & _mask(rn)
        assert both.any(), "the two wirings share no edge; the fixture is wrong"
        assert torch.equal(_chem(wt)[both], _chem(rn)[both])

    def test_the_default_does_not_share_them(
        self,
        wild: dict[str, ConnectomePPOBrain],
        rewired: dict[str, ConnectomePPOBrain],
    ) -> None:
        # The contrast that makes the mode worth having: under edge_order the same shared edges
        # disagree, which is the confound this control removes.
        wt, rn = wild["edge_order"], rewired["edge_order"]
        both = _mask(wt) & _mask(rn)
        assert not torch.equal(_chem(wt)[both], _chem(rn)[both])


class TestPerNeuronFaninSharesEveryMultiset:
    """Every neuron receives the identical multiset of incoming weights."""

    def test_incoming_multiset_matches_across_wirings(
        self,
        wild: dict[str, ConnectomePPOBrain],
        rewired: dict[str, ConnectomePPOBrain],
    ) -> None:
        wt, rn = wild["per_neuron_fanin"], rewired["per_neuron_fanin"]
        w_chem, r_chem, w_mask, r_mask = _chem(wt), _chem(rn), _mask(wt), _mask(rn)
        checked = 0
        for post_j in range(w_chem.shape[1]):
            w_in = torch.sort(w_chem[:, post_j][w_mask[:, post_j]]).values
            r_in = torch.sort(r_chem[:, post_j][r_mask[:, post_j]]).values
            assert w_in.shape == r_in.shape, f"neuron {post_j} in-degree moved under rewiring"
            if w_in.numel():
                assert torch.equal(w_in, r_in), f"neuron {post_j} multiset differs"
                checked += 1
        assert checked > 0, "no neuron had incoming edges; the fixture is wrong"

    def test_the_pairing_still_differs(
        self,
        wild: dict[str, ConnectomePPOBrain],
        rewired: dict[str, ConnectomePPOBrain],
    ) -> None:
        # The mode shares the multiset, not the assignment -- otherwise it would erase the wiring.
        wt, rn = wild["per_neuron_fanin"], rewired["per_neuron_fanin"]
        assert not torch.equal(_chem(wt), _chem(rn))


class TestTheRngStreamIsUntouched:
    """Bitwise identity on both axes a draw mode could disturb."""

    @pytest.mark.parametrize("mode", _MODES)
    def test_periphery_is_identical_across_wirings_at_one_mode(
        self,
        mode: str,
        wild: dict[str, ConnectomePPOBrain],
        rewired: dict[str, ConnectomePPOBrain],
    ) -> None:
        wt, rn = _others(wild[mode]), _others(rewired[mode])
        assert wt.keys() == rn.keys()
        for name, param in wt.items():
            assert torch.equal(param, rn[name]), f"{name} differs across wirings under {mode}"

    @pytest.mark.parametrize("mode", ["dense_mask", "per_neuron_fanin"])
    def test_periphery_is_identical_across_modes_for_one_wiring(
        self,
        mode: str,
        wild: dict[str, ConnectomePPOBrain],
    ) -> None:
        # The second axis: a mode consuming a different number of values from the chemical
        # generator must not move anything drawn from any other stream.
        base, other = _others(wild["edge_order"]), _others(wild[mode])
        assert base.keys() == other.keys()
        for name, param in base.items():
            assert torch.equal(param, other[name]), f"{name} moved under {mode}"

    @pytest.mark.parametrize("mode", _MODES)
    def test_gap_junctions_are_untouched(
        self,
        mode: str,
        wild: dict[str, ConnectomePPOBrain],
    ) -> None:
        base = wild["edge_order"].topology.g_gap.detach()
        assert torch.equal(base, wild[mode].topology.g_gap.detach())

    @pytest.mark.parametrize("mode", _MODES)
    def test_the_edge_set_is_untouched(
        self,
        mode: str,
        wild: dict[str, ConnectomePPOBrain],
    ) -> None:
        # A draw mode chooses values, never which edges exist.
        assert torch.equal(_mask(wild["edge_order"]), _mask(wild[mode]))


class TestTheSharedGeneratorEndsWhereItStarted:
    """A draw mode must not move any stream but its own.

    ``rng`` is shared with the rollout buffer, whose minibatch permutation consumes it. A mode
    taking a different NUMBER of values from it would change PPO's minibatch order as well as the
    weights -- two manipulations under one name, invisible in the initial parameters because it only
    shows up during training. Asserted here rather than argued, which is the failure this suite
    exists to prevent and the one it previously missed.
    """

    def test_the_buffer_shares_the_brain_generator(self) -> None:
        # If this stops being true the rest of the class is checking nothing.
        brain = _brain(_FROZEN)
        assert brain.buffer.rng is brain.rng

    @pytest.mark.parametrize("mode", _MODES)
    def test_every_mode_leaves_the_shared_generator_in_the_same_state(self, mode: str) -> None:
        baseline = _brain(_FROZEN, weight_draw="edge_order")
        other = _brain(_FROZEN, weight_draw=mode)
        expected = baseline.rng.permutation(64)
        actual = other.rng.permutation(64)
        assert (expected == actual).all(), (
            f"{mode} left the shared generator in a different state, so it would move PPO's "
            "minibatch order as well as the chemical weights"
        )


class TestTheUntestedPairingIsRefused:
    def _bad(self) -> ConnectomePPOBrainConfig:
        container = load_simulation_config(str(_FROZEN)).brain
        assert container is not None
        assert isinstance(container.config, ConnectomePPOBrainConfig)
        return container.config.model_copy(
            update={"weight_draw": "dense_mask", "weight_init": "count_scaled"},
        )

    def test_count_scaled_with_a_sharing_mode_raises_on_validation(self) -> None:
        with pytest.raises(ValueError, match="not defined"):
            ConnectomePPOBrainConfig.model_validate(self._bad().model_dump())

    def test_it_also_raises_at_construction(self) -> None:
        # `model_copy(update=...)` skips validators, so a copied config reaches the brain
        # unvalidated. Without the construction-time guard it would run dense_mask semantics while
        # still reporting count_scaled.
        with pytest.raises(ValueError, match="not defined"):
            ConnectomePPOBrain(config=self._bad(), device=DeviceType.CPU)

    def test_count_scaled_with_the_default_draw_is_allowed(self) -> None:
        container = load_simulation_config(str(_FROZEN)).brain
        assert container is not None
        assert isinstance(container.config, ConnectomePPOBrainConfig)
        cfg = container.config.model_copy(update={"weight_init": "count_scaled"})
        assert ConnectomePPOBrainConfig.model_validate(cfg.model_dump()).weight_init == (
            "count_scaled"
        )
