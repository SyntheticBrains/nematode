"""Tests for the 302-neuron classification table."""

from quantumnematode.connectome.model import CellClass
from quantumnematode.connectome.neurons import (
    EXPECTED_NEURON_COUNT,
    NEURON_CLASSIFICATION,
)

VALID_CELL_CLASSES: set[CellClass] = {
    "sensory",
    "interneuron",
    "motor",
    "muscle",
    "pharyngeal",
}


class TestNeuronClassificationTable:
    """Invariants on the static 302-neuron table."""

    def test_table_has_exactly_302_entries(self) -> None:
        """NEURON_CLASSIFICATION has exactly 302 entries."""
        assert len(NEURON_CLASSIFICATION) == EXPECTED_NEURON_COUNT

    def test_every_entry_has_valid_cell_class(self) -> None:
        """Every entry's cell_class is in the allowed CellClass set."""
        for name, (cls, _nt) in NEURON_CLASSIFICATION.items():
            assert cls in VALID_CELL_CLASSES, (
                f"{name}: cell_class={cls!r} not in {VALID_CELL_CLASSES}"
            )

    def test_class_counts_are_logged_for_review(self, capsys) -> None:
        """Per-class counts are printed for logbook review (no band assertion)."""
        counts: dict[str, int] = {}
        for cls, _nt in NEURON_CLASSIFICATION.values():
            counts[cls] = counts.get(cls, 0) + 1
        print(f"NEURON_CLASSIFICATION per-class counts: {counts}")
        # Capture for completeness; test always passes.
        captured = capsys.readouterr()
        assert "per-class counts" in captured.out
