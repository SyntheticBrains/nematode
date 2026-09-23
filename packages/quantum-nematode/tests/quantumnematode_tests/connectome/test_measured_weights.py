"""The vendored measured-weight table: the file, its names, and where it lands on Cook 2019.

Covers the connectome-substrate requirement "Measured synaptic weights from a vendored fitted
model": the vendored file is the one recorded, every name is a known neuron, and coverage on the
Cook 2019 substrate is the documented one.

The coverage figures are pinned rather than recomputed from the same code, because they are what
every later use of the table is sized against: a measured prior reaches only the covered edges, and
nothing onto the body motor neurons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from quantumnematode.brain.arch.connectome_ppo import _MOTOR_CLASSES
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
from quantumnematode.connectome.measured_weights import (
    MEASURED_WEIGHTS_PATH,
    MEASURED_WEIGHTS_SHA256,
    MeasuredWeightsError,
    coverage,
    measured_weights,
    read_measured_weights,
)
from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION

if TYPE_CHECKING:
    from pathlib import Path

_ENTRIES = 2011
_NEURONS = 125
_FULL_SCOPE = 3709
_COVERED = 1049
_HEAD_SCOPE = 1386
_HEAD_SELF_LOOPS = 23
_POSITIVE = 635
_NEGATIVE = 414
_GAP_JUNCTION_ONLY = 265
_NO_CONNECTION = 697
_MOTOR_NEURONS = 39


@pytest.fixture(scope="module")
def cook_coverage():
    """Coverage of the vendored table on the Cook 2019 hermaphrodite."""
    return coverage(measured_weights(), load_cook_2019_hermaphrodite())


class TestTheVendoredFile:
    """The file on disk is the one recorded, byte for byte."""

    def test_the_digest_matches(self) -> None:
        """The bytes hash to the recorded SHA256 — including the upstream CRLF line endings."""
        import hashlib

        assert hashlib.sha256(MEASURED_WEIGHTS_PATH.read_bytes()).hexdigest() == (
            MEASURED_WEIGHTS_SHA256
        )

    def test_a_changed_file_is_refused(self, tmp_path: Path) -> None:
        """A copy with one value altered is refused, not read."""
        altered = tmp_path / MEASURED_WEIGHTS_PATH.name
        altered.write_bytes(MEASURED_WEIGHTS_PATH.read_bytes().replace(b"0.092504", b"0.092505", 1))
        with pytest.raises(MeasuredWeightsError, match="SHA256"):
            read_measured_weights(altered)

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        """A missing table is an error naming the path, not an empty prior."""
        with pytest.raises(MeasuredWeightsError, match="not found"):
            read_measured_weights(tmp_path / "absent.csv")


class TestTheTable:
    """What the table holds."""

    def test_the_entry_and_neuron_counts(self) -> None:
        """2,011 signed entries over 125 neurons, and no diagonal."""
        table = measured_weights()
        assert len(table) == _ENTRIES
        assert len({name for edge in table for name in edge}) == _NEURONS
        assert not [edge for edge in table if edge[0] == edge[1]]

    def test_every_name_is_a_canonical_neuron(self) -> None:
        """Every pre- and post-synaptic name is one of the 302 canonical neurons."""
        names = {name for edge in measured_weights() for name in edge}
        assert names <= set(NEURON_CLASSIFICATION)

    def test_the_table_is_read_only(self) -> None:
        """The mapping cannot be written through, so no caller can alter another's prior."""
        with pytest.raises(TypeError):
            measured_weights()["ADAL", "ADAR"] = 0.0  # type: ignore[index]


class TestCoverageOnCook2019:
    """Where the table lands on the substrate every connectome brain runs on."""

    def test_the_documented_figures(self, cook_coverage) -> None:
        """Covered, head scope and its self-loops, and the entries that land nowhere chemical."""
        assert len(cook_coverage.full_scope) == _FULL_SCOPE
        assert len(cook_coverage.covered) == _COVERED
        assert len(cook_coverage.head_scope) == _HEAD_SCOPE
        assert len(cook_coverage.head_self_loops) == _HEAD_SELF_LOOPS
        assert len(cook_coverage.coverable_head_scope) == _HEAD_SCOPE - _HEAD_SELF_LOOPS
        assert len(cook_coverage.gap_junction_only) == _GAP_JUNCTION_ONLY
        assert len(cook_coverage.no_connection) == _NO_CONNECTION

    def test_every_entry_is_accounted_for_exactly_once(self, cook_coverage) -> None:
        """Covered, gap-junction-only and unconnected entries partition the table."""
        parts = (
            cook_coverage.covered,
            cook_coverage.gap_junction_only,
            cook_coverage.no_connection,
        )
        assert sum(len(p) for p in parts) == _ENTRIES
        assert frozenset().union(*parts) == frozenset(measured_weights())

    def test_the_covered_signs(self, cook_coverage) -> None:
        """635 positive and 414 negative covered values, none zero."""
        values = [measured_weights()[e] for e in cook_coverage.covered]
        assert sum(v > 0 for v in values) == _POSITIVE
        assert sum(v < 0 for v in values) == _NEGATIVE

    def test_nothing_lands_on_the_body_motor_neurons(self, cook_coverage) -> None:
        """The imaging recorded the head, so no covered edge reaches a readout motor neuron."""
        motors = {n for n in NEURON_CLASSIFICATION if n.startswith(_MOTOR_CLASSES)}
        assert len(motors) == _MOTOR_NEURONS
        assert not [e for e in cook_coverage.covered if e[1] in motors]
