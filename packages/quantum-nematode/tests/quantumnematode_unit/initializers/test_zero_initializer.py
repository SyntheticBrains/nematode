import pytest
from quantumnematode.initializers.zero_initializer import ZeroInitializer


def test_zero_initializer_all():
    """Test initializing with all parameters set to zero."""
    initializer = ZeroInitializer()
    num_qubits = 4
    params = initializer.initialize(num_qubits, None)
    assert len(params) == num_qubits
    for v in params.values():
        assert v == 0.0


def test_zero_initializer_with_parameter_list():
    """Test initializing with a list of parameter names."""
    initializer = ZeroInitializer()
    param_names = ["a", "b", "c"]
    params = initializer.initialize(10, param_names)
    assert set(params.keys()) == set(param_names)
    for v in params.values():
        assert v == 0.0


def test_zero_initializer_duplicate_param_raises():
    """Test that initializing with duplicate parameter names raises a ValueError."""
    initializer = ZeroInitializer()
    param_names = ["a", "a"]
    with pytest.raises(ValueError):  # noqa: PT011
        initializer.initialize(2, param_names)


def test_zero_initializer_str():
    """Test the string representation of the ZeroInitializer."""
    assert str(ZeroInitializer()) == "ZeroInitializer(all parameters set to 0.0)"
