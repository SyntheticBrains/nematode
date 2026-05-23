"""Tests for the connectome validators + cross-validation."""

import pytest
from quantumnematode.connectome import (
    ChemicalSynapse,
    Connectome,
    DivergenceReport,
    Neuron,
    cross_validate,
    load_cook_2019_hermaphrodite,
    load_witvliet_2021_adult,
    validate_known_pathways,
    validate_neuron_count,
)

EXPECTED_HERMAPHRODITE_NEURONS = 302
MIN_SHARED_NEURONS_COOK_VS_WITVLIET = 100  # nerve-ring overlap


@pytest.fixture(scope="module")
def cook_2019() -> Connectome:
    """Load the Cook 2019 hermaphrodite connectome once per module."""
    return load_cook_2019_hermaphrodite()


@pytest.fixture(scope="module")
def witvliet_2021() -> Connectome:
    """Load the Witvliet 2021 adult connectome once per module."""
    return load_witvliet_2021_adult()


class TestValidateNeuronCount:
    """Structural neuron-count check."""

    def test_passes_on_cook_2019_herm(self, cook_2019: Connectome) -> None:
        """validate_neuron_count passes on Cook 2019 (302 neurons)."""
        result = validate_neuron_count(cook_2019)
        assert result.passed is True
        assert result.details["actual"] == EXPECTED_HERMAPHRODITE_NEURONS

    def test_fails_on_artificially_broken_connectome(self, cook_2019: Connectome) -> None:
        """validate_neuron_count flags a 301-neuron broken connectome."""
        dropped = next(iter(cook_2019.neurons))
        remaining_neurons = {k: v for k, v in cook_2019.neurons.items() if k != dropped}
        chem = [s for s in cook_2019.chemical_synapses if dropped not in (s.pre, s.post)]
        gj = [g for g in cook_2019.gap_junctions if dropped not in (g.neuron_a, g.neuron_b)]
        broken = Connectome(
            neurons=remaining_neurons,
            chemical_synapses=chem,
            gap_junctions=gj,
            source="test_broken",
            version="test",
        )
        result = validate_neuron_count(broken)
        assert result.passed is False
        assert result.details["actual"] == EXPECTED_HERMAPHRODITE_NEURONS - 1


class TestValidateKnownPathways:
    """Trace classical sensory -> interneuron -> motor pathways."""

    def test_passes_on_cook_2019_herm(self, cook_2019: Connectome) -> None:
        """validate_known_pathways finds at least one canonical pathway in Cook 2019."""
        result = validate_known_pathways(cook_2019)
        assert result.passed is True
        traced = result.details["traced"]
        assert isinstance(traced, list)
        assert len(traced) >= 1

    def test_fails_on_empty_chemical_synapses(self) -> None:
        """validate_known_pathways fails when no chemical synapses exist."""
        empty = Connectome(
            neurons={"X": Neuron(name="X", cell_class="motor")},
            chemical_synapses=[],
            gap_junctions=[],
            source="test_empty",
            version="test",
        )
        result = validate_known_pathways(empty)
        assert result.passed is False
        traced = result.details["traced"]
        assert isinstance(traced, list)
        assert len(traced) == 0


class TestCrossValidate:
    """Diff Cook 2019 hermaphrodite vs Witvliet 2021 adult dataset 8."""

    def test_returns_divergence_report(
        self,
        cook_2019: Connectome,
        witvliet_2021: Connectome,
    ) -> None:
        """cross_validate returns a DivergenceReport instance."""
        report = cross_validate(cook_2019, witvliet_2021)
        assert isinstance(report, DivergenceReport)

    def test_shared_neurons_above_threshold(
        self,
        cook_2019: Connectome,
        witvliet_2021: Connectome,
    ) -> None:
        """Cook vs Witvliet share > 100 neurons (the nerve-ring overlap)."""
        report = cross_validate(cook_2019, witvliet_2021)
        assert report.shared_neurons > MIN_SHARED_NEURONS_COOK_VS_WITVLIET

    def test_agreement_set_is_non_empty(
        self,
        cook_2019: Connectome,
        witvliet_2021: Connectome,
    ) -> None:
        """Cook vs Witvliet agree on at least some shared (pre, post) pairs."""
        report = cross_validate(cook_2019, witvliet_2021)
        assert report.shared_pairs_agreement > 0

    def test_records_sources(
        self,
        cook_2019: Connectome,
        witvliet_2021: Connectome,
    ) -> None:
        """DivergenceReport records both input connectomes' source identifiers."""
        report = cross_validate(cook_2019, witvliet_2021)
        assert report.primary_source == cook_2019.source
        assert report.secondary_source == witvliet_2021.source

    def test_weight_divergence_summary_has_stats(
        self,
        cook_2019: Connectome,
        witvliet_2021: Connectome,
    ) -> None:
        """weight_divergence_summary contains n_pairs + mean + median when n>0."""
        report = cross_validate(cook_2019, witvliet_2021)
        assert "n_pairs" in report.weight_divergence_summary
        if report.weight_divergence_summary["n_pairs"] > 0:
            assert "mean" in report.weight_divergence_summary
            assert "median" in report.weight_divergence_summary

    def test_synthetic_full_agreement(self) -> None:
        """Two identical connectomes produce all-agreement (no diff)."""
        n = {
            "A": Neuron(name="A", cell_class="sensory"),
            "B": Neuron(name="B", cell_class="interneuron"),
            "C": Neuron(name="C", cell_class="motor"),
        }
        syn = [
            ChemicalSynapse(pre="A", post="B", weight=5),
            ChemicalSynapse(pre="B", post="C", weight=3),
        ]
        c1 = Connectome(
            neurons=n,
            chemical_synapses=syn,
            gap_junctions=[],
            source="c1",
            version="v",
        )
        c2 = Connectome(
            neurons=n,
            chemical_synapses=syn,
            gap_junctions=[],
            source="c2",
            version="v",
        )
        report = cross_validate(c1, c2)
        assert report.shared_neurons == 3
        assert report.shared_pairs_disagreement == 0
        assert len(report.primary_only_pairs) == 0
        assert len(report.secondary_only_pairs) == 0
        if report.weight_divergence_summary["n_pairs"] > 0:
            assert report.weight_divergence_summary["mean"] == 1.0
            assert report.weight_divergence_summary["median"] == 1.0
