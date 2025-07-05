import numpy as np
import pytest
from quantumnematode.initializers.random_initializer import (
    RandomPiUniformInitializer,
    RandomSmallUniformInitializer,
)


def test_random_pi_uniform_initializer_all():
    """Test initializing with uniform values in the range [-pi, pi]."""
    initializer = RandomPiUniformInitializer()
    num_qubits = 5
    params = initializer.initialize(num_qubits, None)
    assert len(params) == num_qubits
    for v in params.values():
        assert -np.pi <= v <= np.pi


def test_random_small_uniform_initializer_all():
    """Test initializing with small uniform values."""
    initializer = RandomSmallUniformInitializer()
    num_qubits = 5
    params = initializer.initialize(num_qubits, None)
    assert len(params) == num_qubits
    for v in params.values():
        assert -0.1 <= v <= 0.1


@pytest.mark.parametrize(
    ("initializer_cls", "low", "high"),
    [
        (RandomPiUniformInitializer, -np.pi, np.pi),
        (RandomSmallUniformInitializer, -0.1, 0.1),
    ],
)
def test_random_initializer_with_parameter_list(initializer_cls, low, high):
    """Test initializing with a list of parameter names."""
    initializer = initializer_cls()
    param_names = ["a", "b", "c"]
    params = initializer.initialize(10, param_names)
    assert set(params.keys()) == set(param_names)
    for v in params.values():
        assert low <= v <= high


def test_random_initializer_duplicate_param_raises():
    """Test that initializing with duplicate parameter names raises a ValueError."""
    initializer = RandomPiUniformInitializer()
    param_names = ["a", "a"]
    with pytest.raises(ValueError):  # noqa: PT011
        initializer.initialize(2, param_names)


def test_str_methods():
    """Test the string representation of initializers."""
    assert str(RandomPiUniformInitializer()) == "RandomPiUniformInitializer(range=[-pi, pi])"
    assert (
        str(RandomSmallUniformInitializer()) == "RandomSmallUniformInitializer(range=[-0.1, 0.1])"
    )
