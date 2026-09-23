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
    critic = getattr(getattr(brain, "rule", None), "critic", None)
    if critic is not None:
        out |= {f"critic.{n}": p.detach() for n, p in critic.named_parameters()}
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
    @pytest.mark.parametrize("path", [_WILD, _REWIRED], ids=["wild", "rewired"])
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
        ],
        ids=["scale-under-random", "scale-under-signs", "atlas", "draw", "count-init"],
    )
    def test_refused_at_validation_and_at_construction(self, overrides: dict[str, object]) -> None:
        """Each pairing is refused by the validator, and again when validation was skipped."""
        bad = self._config(**overrides)
        with pytest.raises(ValueError, match=r"never read|not defined"):
            ConnectomePPOBrainConfig.model_validate(bad.model_dump())
        with pytest.raises(ValueError, match=r"never read|not defined"):
            ConnectomePPOBrain(config=bad, device=DeviceType.CPU)

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
