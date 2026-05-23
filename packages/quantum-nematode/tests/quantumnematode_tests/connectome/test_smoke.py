"""Tests for the connectome forward-pass smoke check."""

import numpy as np
import pytest
from quantumnematode.connectome import (
    Connectome,
    Neuron,
    load_cook_2019_hermaphrodite,
)
from quantumnematode.connectome.smoke import (
    _DEGENERATE_VARIANCE_THRESHOLD,
    run_forward_pass,
)


@pytest.fixture(scope="module")
def cook_2019() -> Connectome:
    """Load the Cook 2019 hermaphrodite connectome once per module."""
    return load_cook_2019_hermaphrodite()


class TestForwardPassOnCook2019:
    """Verify the smoke forward pass produces finite + non-degenerate output."""

    def test_output_is_finite(self, cook_2019: Connectome) -> None:
        """Forward pass returns finite values (no NaN/Inf)."""
        output = run_forward_pass(cook_2019, seed=0)
        assert np.all(np.isfinite(output))

    def test_output_shape_matches_motor_neuron_count(self, cook_2019: Connectome) -> None:
        """Output has one entry per motor neuron in the connectome."""
        output = run_forward_pass(cook_2019, seed=0)
        motor_count = sum(1 for n in cook_2019.neurons.values() if n.cell_class == "motor")
        assert output.shape == (motor_count,)

    def test_output_has_non_degenerate_variance(self, cook_2019: Connectome) -> None:
        """Output variance is above the degeneracy threshold."""
        output = run_forward_pass(cook_2019, seed=0)
        assert float(output.var()) >= _DEGENERATE_VARIANCE_THRESHOLD

    def test_seed_determinism(self, cook_2019: Connectome) -> None:
        """Same seed produces bit-identical output."""
        a = run_forward_pass(cook_2019, seed=42)
        b = run_forward_pass(cook_2019, seed=42)
        assert np.allclose(a, b)

    def test_different_seeds_produce_different_outputs(
        self,
        cook_2019: Connectome,
    ) -> None:
        """Different seeds produce different outputs (vanishingly unlikely tie)."""
        a = run_forward_pass(cook_2019, seed=0)
        b = run_forward_pass(cook_2019, seed=1)
        assert not np.allclose(a, b)


class TestSanityGuard:
    """The smoke pass raises on a deliberately-emptied connectome."""

    def test_raises_on_empty_chemical_synapses(self) -> None:
        """run_forward_pass raises ValueError on empty chemical synapses."""
        empty = Connectome(
            neurons={"X": Neuron(name="X", cell_class="motor")},
            chemical_synapses=[],
            gap_junctions=[],
            source="test_empty",
            version="test",
        )
        with pytest.raises(ValueError, match="zero chemical synapses"):
            run_forward_pass(empty)
