"""Tests for the connectome data-model invariants."""

import pytest
from pydantic import ValidationError
from quantumnematode.connectome import (
    ChemicalSynapse,
    Connectome,
    GapJunction,
    Neuron,
    load_cook_2019_hermaphrodite,
)


class TestModelFieldValidation:
    """Pydantic field-level invariants."""

    def test_chemical_synapse_rejects_zero_weight(self) -> None:
        """ChemicalSynapse with weight=0 raises ValidationError (gt=0)."""
        with pytest.raises(ValidationError):
            ChemicalSynapse(pre="ASEL", post="AIYL", weight=0)

    def test_chemical_synapse_rejects_negative_weight(self) -> None:
        """ChemicalSynapse with weight<0 raises ValidationError."""
        with pytest.raises(ValidationError):
            ChemicalSynapse(pre="ASEL", post="AIYL", weight=-1)

    def test_gap_junction_accepts_zero_weight(self) -> None:
        """GapJunction with weight=0 is allowed (ge=0)."""
        gj = GapJunction(neuron_a="ASEL", neuron_b="AVAL", weight=0)
        assert gj.weight == 0

    def test_gap_junction_rejects_negative_weight(self) -> None:
        """GapJunction with weight<0 raises ValidationError."""
        with pytest.raises(ValidationError):
            GapJunction(neuron_a="ASEL", neuron_b="AVAL", weight=-1)

    def test_gap_junction_rejects_non_canonical_pair_order(self) -> None:
        """GapJunction requires neuron_a < neuron_b lexicographically."""
        with pytest.raises(ValidationError):
            GapJunction(neuron_a="AVAL", neuron_b="ASEL", weight=5)

    def test_gap_junction_rejects_self_loop(self) -> None:
        """GapJunction with neuron_a == neuron_b raises (not strictly less-than)."""
        with pytest.raises(ValidationError):
            GapJunction(neuron_a="AVAL", neuron_b="AVAL", weight=1)


class TestConnectomeIntegrity:
    """Connectome model-level invariants."""

    def test_rejects_orphan_chemical_synapse(self) -> None:
        """Connectome rejects a chemical synapse referencing an unknown neuron."""
        with pytest.raises(ValidationError):
            Connectome(
                neurons={"AVAL": Neuron(name="AVAL", cell_class="interneuron")},
                chemical_synapses=[ChemicalSynapse(pre="AVAL", post="GHOST", weight=1)],
                gap_junctions=[],
                source="test",
                version="test",
            )

    def test_rejects_orphan_gap_junction(self) -> None:
        """Connectome rejects a gap junction referencing an unknown neuron."""
        with pytest.raises(ValidationError):
            Connectome(
                neurons={"AVAL": Neuron(name="AVAL", cell_class="interneuron")},
                chemical_synapses=[],
                gap_junctions=[GapJunction(neuron_a="AVAL", neuron_b="GHOST", weight=1)],
                source="test",
                version="test",
            )

    def test_minimal_connectome_validates(self) -> None:
        """A two-neuron connectome with one of each edge type validates."""
        c = Connectome(
            neurons={
                "A": Neuron(name="A", cell_class="sensory"),
                "B": Neuron(name="B", cell_class="motor"),
            },
            chemical_synapses=[ChemicalSynapse(pre="A", post="B", weight=1)],
            gap_junctions=[GapJunction(neuron_a="A", neuron_b="B", weight=1)],
            source="test",
            version="test",
        )
        assert len(c.neurons) == 2
        assert len(c.chemical_synapses) == 1
        assert len(c.gap_junctions) == 1


class TestDualEdgeCase:
    """Pair connected by BOTH chemical + gap junction must appear as two distinct entries.

    AVAL <-> AVDL is the canonical dual-edge example in Cook 2019: both
    chemical directions AND a gap junction.
    """

    @pytest.fixture(scope="class")
    def connectome(self) -> Connectome:
        """Load the Cook 2019 hermaphrodite connectome once per class."""
        return load_cook_2019_hermaphrodite()

    def test_aval_avdl_has_chemical_synapses(self, connectome: Connectome) -> None:
        """AVAL <-> AVDL has at least one chemical synapse."""
        chem = [s for s in connectome.chemical_synapses if {s.pre, s.post} == {"AVAL", "AVDL"}]
        assert len(chem) >= 1

    def test_aval_avdl_has_gap_junction(self, connectome: Connectome) -> None:
        """AVAL <-> AVDL has exactly one gap junction in canonical form."""
        gj = [g for g in connectome.gap_junctions if {g.neuron_a, g.neuron_b} == {"AVAL", "AVDL"}]
        assert len(gj) == 1

    def test_dual_edges_are_separate_entries(self, connectome: Connectome) -> None:
        """ChemicalSynapse and GapJunction are different types, not merged."""
        chem = [s for s in connectome.chemical_synapses if {s.pre, s.post} == {"AVAL", "AVDL"}]
        gj = [g for g in connectome.gap_junctions if {g.neuron_a, g.neuron_b} == {"AVAL", "AVDL"}]
        for s in chem:
            assert isinstance(s, ChemicalSynapse)
        for g in gj:
            assert isinstance(g, GapJunction)
