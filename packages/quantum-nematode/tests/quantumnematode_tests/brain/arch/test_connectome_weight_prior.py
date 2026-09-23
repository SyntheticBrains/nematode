"""A measured prior on the chemical weights: what it moves, where it lands, what it leaves alone.

Covers the connectome-ppo-brain requirement "A measured prior for the chemical weights": the default
is bit-identical; a prior changes covered chemical edges and nothing else; the shared generator is
left where it was; the rewired null receives each neuron's wild-type values; a multiplier no prior
reads is refused; an untested pairing is refused.

Every property is asserted against constructed brains rather than argued from how the loop is
written. The shared-generator check is the one a construction test would miss: the buffer that
permutes PPO's minibatches draws from the same generator the weight draw does, so a prior that
took a different number of values from it would move training as well as the weights.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
from quantumnematode.connectome.measured_weights import coverage, measured_weights
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[6]
_ARMS = _REPO_ROOT / "configs" / "scenarios" / "foraging"
_WILD = _ARMS / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350.yml"
_REWIRED = (
    _ARMS / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_rewired_null.yml"
)
_FANIN_WILD = _ARMS / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_fanin.yml"
_FANIN_REWIRED = (
    _ARMS
    / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_rewired_null_fanin.yml"
)

_SEED = 17
_MEASURED = ("measured", "measured_signs", "measured_shuffled")
_DRAWN = "w_chem"


def _brain(path: Path, **overrides: object) -> ConnectomePPOBrain:
    container = load_simulation_config(str(path)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    cfg = container.config.model_copy(update={"seed": _SEED, **overrides})
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _at(brain: ConnectomePPOBrain, pre: str, post: str) -> float:
    idx = brain.topology._idx
    return float(brain.topology.w_chem.detach()[idx[pre], idx[post]])


def _others(brain: ConnectomePPOBrain) -> dict[str, torch.Tensor]:
    """Every parameter a prior does not claim to touch, topology and critic alike."""
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


def _edges(brain: ConnectomePPOBrain) -> list[tuple[str, str]]:
    names = brain.topology.neuron_names
    mask = brain.topology.m_chem.detach().numpy()
    return [(names[i], names[j]) for i, j in zip(*np.nonzero(mask), strict=True)]


@pytest.fixture(scope="module")
def covered() -> list[tuple[str, str]]:
    """Return the wild type's covered chemical edges, sorted."""
    return sorted(coverage(measured_weights(), load_cook_2019_hermaphrodite()).covered)


@pytest.fixture(scope="module")
def wild() -> dict[str, ConnectomePPOBrain]:
    """Wild-type brains at one seed, one per prior."""
    return {p: _brain(_WILD, weight_prior=p) for p in ("random", *_MEASURED)}


@pytest.fixture(scope="module")
def rewired() -> dict[str, ConnectomePPOBrain]:
    """Rewired-null brains at one seed, one per prior."""
    return {p: _brain(_REWIRED, weight_prior=p) for p in ("random", *_MEASURED)}


def _in_degree(brain: ConnectomePPOBrain, post: str) -> int:
    return int(brain.topology.m_chem.detach()[:, brain.topology._idx[post]].sum())


class TestTheDefaultIsUnchanged:
    def test_an_explicit_random_prior_is_bit_identical_to_the_default(self) -> None:
        """Stating the default changes nothing, on any parameter."""
        default = _brain(_WILD)
        explicit = _brain(_WILD, weight_prior="random")
        a = dict(default.topology.named_parameters())
        b = dict(explicit.topology.named_parameters())
        for name, param in a.items():
            assert torch.equal(param, b[name]), name


class TestOnlyTheCoveredChemicalEdgesMove:
    @pytest.mark.parametrize("prior", _MEASURED)
    @pytest.mark.parametrize("wiring", ["wild", "rewired"])
    def test_every_other_parameter_is_identical(
        self,
        prior: str,
        wiring: str,
        wild: dict[str, ConnectomePPOBrain],
        rewired: dict[str, ConnectomePPOBrain],
    ) -> None:
        """A prior moves the chemical weights and no other tensor, critic included."""
        arms = wild if wiring == "wild" else rewired
        base, other = _others(arms["random"]), _others(arms[prior])
        # These are PPO arms, so the critic must be in the comparison rather than silently absent.
        assert any(name.startswith("critic.") for name in base)
        assert base.keys() == other.keys()
        for name, param in base.items():
            assert torch.equal(param, other[name]), f"{name} moved under {prior}"

    @pytest.mark.parametrize("prior", _MEASURED)
    def test_uncovered_edges_keep_their_draw(
        self,
        prior: str,
        wild: dict[str, ConnectomePPOBrain],
        covered: list[tuple[str, str]],
    ) -> None:
        """On the wild type every edge the table does not cover is the random build's."""
        base = wild["random"]
        uncovered = set(_edges(base)) - set(covered)
        for pre, post in uncovered:
            assert _at(wild[prior], pre, post) == _at(base, pre, post)


class TestWhatEachPriorPlaces:
    def test_measured_places_each_value_on_the_draws_scale_times_one_constant(
        self,
        wild: dict[str, ConnectomePPOBrain],
        covered: list[tuple[str, str]],
    ) -> None:
        """Every covered edge is its fitted value times its 1/sqrt(in-degree) times one constant."""
        table = measured_weights()
        brain = wild["measured"]
        ratios = [
            _at(brain, pre, post) * np.sqrt(_in_degree(brain, post)) / table[pre, post]
            for pre, post in covered
        ]
        assert min(ratios) == pytest.approx(max(ratios), rel=1e-5)
        assert min(ratios) > 0

    def test_at_the_default_multiplier_the_magnitude_is_the_draws(
        self,
        wild: dict[str, ConnectomePPOBrain],
        covered: list[tuple[str, str]],
    ) -> None:
        """Over the covered edges, the placed RMS equals the draw's EXPECTED RMS on those edges.

        The expectation, not the realised draw at one seed, because that is what does not depend on
        the seed. Normalising the fitted values alone missed this by about 27%: the large values sit
        on neurons with few inputs, whose per-neuron scale is large.
        """
        brain = wild["measured"]
        placed = np.array([_at(brain, pre, post) for pre, post in covered])
        expected = np.array([1.0 / _in_degree(brain, post) for _, post in covered])
        assert float(np.sqrt(np.mean(placed**2))) == pytest.approx(
            float(np.sqrt(np.mean(expected))),
            rel=1e-5,
        )

    def test_measured_signs_keeps_the_draws_magnitude(
        self,
        wild: dict[str, ConnectomePPOBrain],
        covered: list[tuple[str, str]],
    ) -> None:
        """Magnitudes are the random build's; signs are the table's."""
        table = measured_weights()
        for pre, post in covered:
            value = _at(wild["measured_signs"], pre, post)
            assert abs(value) == abs(_at(wild["random"], pre, post))
            assert np.sign(value) == np.sign(table[pre, post])

    def test_measured_shuffled_keeps_the_values_and_moves_them(
        self,
        wild: dict[str, ConnectomePPOBrain],
        covered: list[tuple[str, str]],
    ) -> None:
        """Unscaled, the shuffled arm holds the measured arm's values, on other edges."""
        measured, shuffled = wild["measured"], wild["measured_shuffled"]

        def unscaled(brain: ConnectomePPOBrain) -> list[float]:
            return [
                _at(brain, pre, post) * np.sqrt(_in_degree(brain, post)) for pre, post in covered
            ]

        a, b = unscaled(measured), unscaled(shuffled)
        assert sorted(a) == pytest.approx(sorted(b), rel=1e-5)
        assert a != pytest.approx(b, rel=1e-5)

    def test_the_multiplier_scales_covered_edges_linearly(
        self,
        wild: dict[str, ConnectomePPOBrain],
        covered: list[tuple[str, str]],
    ) -> None:
        """Doubling the multiplier doubles every covered edge and leaves the rest alone."""
        doubled = _brain(_WILD, weight_prior="measured", measured_weight_scale=2.0)
        for pre, post in covered:
            assert _at(doubled, pre, post) == pytest.approx(
                2.0 * _at(wild["measured"], pre, post),
                rel=1e-6,
            )
        uncovered = set(_edges(doubled)) - set(covered)
        for pre, post in uncovered:
            assert _at(doubled, pre, post) == _at(wild["random"], pre, post)


class TestTheRewiredNullReceivesEachNeuronsWildTypeValues:
    @pytest.mark.parametrize("prior", _MEASURED)
    def test_first_k_edges_carry_the_wild_type_values_and_the_rest_the_draw(
        self,
        prior: str,
        wild: dict[str, ConnectomePPOBrain],
        rewired: dict[str, ConnectomePPOBrain],
        covered: list[tuple[str, str]],
    ) -> None:
        """Per neuron: wild-type values, in wild-type pre order, on its first k edges."""
        wt, null, null_random = wild[prior], rewired[prior], rewired["random"]
        by_post: dict[str, list[str]] = {}
        for pre, post in covered:
            by_post.setdefault(post, []).append(pre)
        null_incoming: dict[str, list[str]] = {}
        for pre, post in _edges(null):
            null_incoming.setdefault(post, []).append(pre)

        for post, wt_pres in by_post.items():
            partners = sorted(null_incoming[post])
            k = len(wt_pres)
            for wt_pre, null_pre in zip(sorted(wt_pres), partners[:k], strict=True):
                got, want = _at(null, null_pre, post), _at(wt, wt_pre, post)
                if prior == "measured_signs":
                    assert np.sign(got) == np.sign(want)
                    assert abs(got) == abs(_at(null_random, null_pre, post))
                else:
                    # In-degree is preserved, so the per-neuron scale is the same on both graphs.
                    assert got == pytest.approx(want, rel=1e-6)
            for null_pre in partners[k:]:
                assert _at(null, null_pre, post) == _at(null_random, null_pre, post)


class TestTheSharedGeneratorEndsWhereItStarted:
    def test_the_buffer_shares_the_brain_generator(self) -> None:
        """If this stops being true the class below is checking nothing."""
        brain = _brain(_WILD)
        assert brain.buffer.rng is brain.rng

    @pytest.mark.parametrize("prior", _MEASURED)
    @pytest.mark.parametrize(
        "path",
        [_WILD, _REWIRED, _FANIN_WILD, _FANIN_REWIRED],
        ids=["wild", "rewired", "fanin-wild", "fanin-rewired"],
    )
    def test_every_prior_leaves_the_shared_generator_in_the_same_state(
        self,
        prior: str,
        path: Path,
    ) -> None:
        """A prior must not change what PPO's minibatch sampler draws next."""
        baseline = _brain(path, weight_prior="random")
        other = _brain(path, weight_prior=prior)
        assert (baseline.rng.permutation(64) == other.rng.permutation(64)).all()


class TestRefusals:
    @staticmethod
    def _config(**overrides: object) -> ConnectomePPOBrainConfig:
        container = load_simulation_config(str(_WILD)).brain
        assert container is not None
        assert isinstance(container.config, ConnectomePPOBrainConfig)
        return container.config.model_copy(update={"seed": _SEED, **overrides})

    @pytest.mark.parametrize(
        "overrides",
        [
            {"weight_prior": "random", "measured_weight_scale": 2.0},
            {"weight_prior": "measured_signs", "measured_weight_scale": 2.0},
            {"weight_prior": "measured", "synapse_signs": "atlas"},
            {"weight_prior": "measured", "weight_draw": "dense_mask"},
            {"weight_prior": "measured", "weight_init": "count_scaled"},
            {"weight_prior": "measured", "measured_weight_scale": -1.0},
            {"weight_prior": "measured", "measured_weight_scale": 0.0},
        ],
        ids=[
            "scale-under-random",
            "scale-under-signs",
            "atlas",
            "draw",
            "count-init",
            "negative-scale",
            "zero-scale",
        ],
    )
    def test_refused_at_validation_and_at_construction(self, overrides: dict[str, object]) -> None:
        """Each pairing is refused by the validator, and again when validation was skipped."""
        bad = self._config(**overrides)
        with pytest.raises(ValueError, match=r"never read|not defined|greater than 0|positive"):
            ConnectomePPOBrainConfig.model_validate(bad.model_dump())
        with pytest.raises(ValueError, match=r"never read|not defined|positive"):
            ConnectomePPOBrain(config=bad, device=DeviceType.CPU)

    def test_the_fan_in_draw_is_accepted(self) -> None:
        """Every measured prior validates, and constructs, under the per-neuron fan-in draw."""
        for prior in _MEASURED:
            cfg = self._config(weight_prior=prior, weight_draw="per_neuron_fanin")
            ConnectomePPOBrainConfig.model_validate(cfg.model_dump())
            ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)

    def test_a_multiplier_is_accepted_where_it_is_read(self) -> None:
        """The measured and shuffled priors read it, so a non-default value validates."""
        for prior in ("measured", "measured_shuffled"):
            ConnectomePPOBrainConfig.model_validate(
                self._config(weight_prior=prior, measured_weight_scale=2.0).model_dump(),
            )


def test_the_training_state_records_the_prior() -> None:
    """A saved run says which prior and multiplier built it, and which draw."""
    brain = _brain(_WILD, weight_prior="measured", measured_weight_scale=1.5)
    state = brain.get_weight_components()["training_state"].state
    assert state["weight_prior"] == "measured"
    assert state["measured_weight_scale"] == 1.5
    assert state["weight_draw"] == "edge_order"


@pytest.fixture(scope="module")
def wild_fanin() -> dict[str, ConnectomePPOBrain]:
    """Wild-type brains under the per-neuron fan-in draw, one per prior."""
    return {p: _brain(_FANIN_WILD, weight_prior=p) for p in ("random", *_MEASURED)}


@pytest.fixture(scope="module")
def rewired_fanin() -> dict[str, ConnectomePPOBrain]:
    """Rewired-null brains under the per-neuron fan-in draw, one per prior."""
    return {p: _brain(_FANIN_REWIRED, weight_prior=p) for p in ("random", *_MEASURED)}


def _incoming(brain: ConnectomePPOBrain) -> dict[str, list[str]]:
    """Each post-synaptic neuron's pre-synaptic partners, sorted."""
    out: dict[str, list[str]] = {}
    for pre, post in _edges(brain):
        out.setdefault(post, []).append(pre)
    return {post: sorted(pres) for post, pres in out.items()}


class TestUnderTheFanInDraw:
    """Covers "Under the fan-in draw every neuron keeps its wild-type multiset"."""

    def test_the_configs_carry_the_fan_in_draw(self) -> None:
        """If these parents stop using the fan-in draw, the class below checks the wrong mode."""
        for path in (_FANIN_WILD, _FANIN_REWIRED):
            assert _brain(path).config.weight_draw == "per_neuron_fanin"

    @pytest.mark.parametrize("prior", _MEASURED)
    @pytest.mark.parametrize("wiring", ["wild", "rewired"])
    def test_every_other_parameter_is_identical(
        self,
        prior: str,
        wiring: str,
        wild_fanin: dict[str, ConnectomePPOBrain],
        rewired_fanin: dict[str, ConnectomePPOBrain],
    ) -> None:
        """A prior moves the chemical weights and no other tensor, critic included."""
        arms = wild_fanin if wiring == "wild" else rewired_fanin
        base, other = _others(arms["random"]), _others(arms[prior])
        assert any(name.startswith("critic.") for name in base)
        assert base.keys() == other.keys()
        for name, param in base.items():
            assert torch.equal(param, other[name]), f"{name} moved under {prior}"

    @pytest.mark.parametrize("prior", _MEASURED)
    def test_the_wild_types_uncovered_edges_are_its_random_build(
        self,
        prior: str,
        wild_fanin: dict[str, ConnectomePPOBrain],
        covered: list[tuple[str, str]],
    ) -> None:
        """On the wild type the fan-in draw is untouched wherever the table does not reach."""
        base = wild_fanin["random"]
        for pre, post in set(_edges(base)) - set(covered):
            assert _at(wild_fanin[prior], pre, post) == _at(base, pre, post)

    @pytest.mark.parametrize("prior", _MEASURED)
    def test_the_null_takes_covered_values_first_then_the_uncovered_draws(
        self,
        prior: str,
        wild_fanin: dict[str, ConnectomePPOBrain],
        rewired_fanin: dict[str, ConnectomePPOBrain],
        covered: list[tuple[str, str]],
    ) -> None:
        """Per neuron: the wild type's covered values, then its uncovered ones, each in its order.

        Exact equality throughout, including the covered values under the sign-only prior: there the
        null places the wild type's own magnitude, not the draw that happens to land on its edge.
        """
        wt, null = wild_fanin[prior], rewired_fanin[prior]
        wt_in, null_in = _incoming(wt), _incoming(null)
        covered_set = set(covered)
        checked = 0
        for post, wt_pres in wt_in.items():
            first = [pre for pre in wt_pres if (pre, post) in covered_set]
            rest = [pre for pre in wt_pres if (pre, post) not in covered_set]
            want = [_at(wt, pre, post) for pre in (*first, *rest)]
            got = [_at(null, pre, post) for pre in null_in[post]]
            if prior == "measured_signs":
                assert got == want, post
            else:
                assert got == pytest.approx(want, rel=1e-6), post
            checked += bool(first)
        assert checked > 0

    @pytest.mark.parametrize("prior", ["random", *_MEASURED])
    def test_every_neuron_keeps_its_wild_type_multiset(
        self,
        prior: str,
        wild_fanin: dict[str, ConnectomePPOBrain],
        rewired_fanin: dict[str, ConnectomePPOBrain],
    ) -> None:
        """The property the fan-in draw exists for, under every prior including the default."""
        wt, null = wild_fanin[prior], rewired_fanin[prior]
        wt_in, null_in = _incoming(wt), _incoming(null)
        assert wt_in.keys() == null_in.keys()
        for post, wt_pres in wt_in.items():
            a = sorted(_at(wt, pre, post) for pre in wt_pres)
            b = sorted(_at(null, pre, post) for pre in null_in[post])
            assert a == pytest.approx(b, rel=1e-6), post

    def test_under_the_edge_order_draw_the_multiset_is_not_kept(
        self,
        wild: dict[str, ConnectomePPOBrain],
        rewired: dict[str, ConnectomePPOBrain],
    ) -> None:
        """The contrast that makes the test above mean something: edge order shares no multiset."""
        wt, null = wild["measured"], rewired["measured"]
        wt_in, null_in = _incoming(wt), _incoming(null)
        differ = sum(
            sorted(_at(wt, pre, post) for pre in pres)
            != pytest.approx(sorted(_at(null, pre, post) for pre in null_in[post]), rel=1e-6)
            for post, pres in wt_in.items()
        )
        assert differ > 0


class TestTheShuffleHasItsOwnStream:
    """Covers "The shuffle does not depend on the draw"."""

    def test_the_permutation_is_the_same_under_either_draw(
        self,
        wild: dict[str, ConnectomePPOBrain],
        wild_fanin: dict[str, ConnectomePPOBrain],
        covered: list[tuple[str, str]],
    ) -> None:
        """Covered values under the shuffled prior do not depend on how the rest were drawn."""
        for pre, post in covered:
            assert _at(wild["measured_shuffled"], pre, post) == _at(
                wild_fanin["measured_shuffled"],
                pre,
                post,
            )

    def test_the_shuffle_is_not_drawn_from_the_draw_generators_stream(
        self,
        wild: dict[str, ConnectomePPOBrain],
        covered: list[tuple[str, str]],
    ) -> None:
        """A generator at the bare run seed is the draw's stream; the shuffle must not be it.

        Fails if the shuffle is given ``get_rng(seed)``, which is what the draw generator is.
        """
        from quantumnematode.brain.arch.connectome_ppo import measured_prior_assignment
        from quantumnematode.utils.seeding import get_rng

        cook = load_cook_2019_hermaphrodite()
        on_draw_stream = measured_prior_assignment(
            "measured_shuffled",
            1.0,
            cook,
            cook,
            rewired=False,
            shuffle_rng=get_rng(_SEED),
        )
        brain = wild["measured_shuffled"]
        placed = [_at(brain, pre, post) * np.sqrt(_in_degree(brain, post)) for pre, post in covered]
        drawn_stream = [on_draw_stream.values[edge] for edge in covered]
        assert placed != pytest.approx(drawn_stream, rel=1e-5)
