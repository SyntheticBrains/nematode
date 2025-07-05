import numpy as np
import pytest
from quantumnematode.optimizers.learning_rate import (
    AdamLearningRate,
    DecayType,
    DynamicLearningRate,
    PerformanceBasedLearningRate,
)


def test_dynamic_learning_rate_inverse_time():
    """Test inverse time decay learning rate."""
    lr = DynamicLearningRate(
        initial_learning_rate=1.0,
        decay_rate=0.5,
        decay_type=DecayType.INVERSE_TIME,
    )
    rates = [lr.get_learning_rate() for _ in range(3)]
    assert rates[0] == 1.0
    assert rates[1] == pytest.approx(1.0 / (1 + 0.5 * 1))
    assert rates[2] == pytest.approx(1.0 / (1 + 0.5 * 2))


def test_dynamic_learning_rate_exponential():
    """Test exponential decay learning rate."""
    lr = DynamicLearningRate(
        initial_learning_rate=1.0,
        decay_rate=0.5,
        decay_type=DecayType.EXPONENTIAL,
    )
    rates = [lr.get_learning_rate() for _ in range(3)]
    assert rates[0] == 1.0
    assert rates[1] == pytest.approx(1.0 * np.exp(-0.5 * 1))
    assert rates[2] == pytest.approx(1.0 * np.exp(-0.5 * 2))


def test_dynamic_learning_rate_step():
    """Test step decay learning rate."""
    lr = DynamicLearningRate(
        initial_learning_rate=1.0,
        decay_type=DecayType.STEP,
        decay_factor=0.5,
        step_size=2,
    )
    rates = [lr.get_learning_rate() for _ in range(5)]
    assert rates[0] == 1.0
    assert rates[2] == 0.5
    assert rates[4] == 0.25


def test_dynamic_learning_rate_polynomial():
    """Test polynomial decay learning rate."""
    lr = DynamicLearningRate(
        initial_learning_rate=1.0,
        decay_type=DecayType.POLYNOMIAL,
        max_steps=4,
        power=2,
    )
    rates = [lr.get_learning_rate() for _ in range(5)]
    assert rates[0] == 1.0
    assert rates[4] == 0.0


def test_dynamic_learning_rate_cosine():
    """Test cosine decay learning rate."""
    lr = DynamicLearningRate(
        initial_learning_rate=1.0,
        decay_type=DecayType.COSINE,
        max_steps=2,
        min_lr=0.1,
    )
    rates = [lr.get_learning_rate() for _ in range(3)]
    assert rates[0] == pytest.approx(1.0)
    assert rates[1] == pytest.approx(0.55)
    assert rates[2] == pytest.approx(0.1)


def test_dynamic_learning_rate_str():
    """Test string representation of DynamicLearningRate."""
    lr = DynamicLearningRate()
    s = str(lr)
    assert "DynamicLearningRate(" in s


def test_adam_learning_rate_basic():
    """Test basic Adam learning rate calculation."""
    adam = AdamLearningRate(initial_learning_rate=0.1)
    grads = [0.5, -0.5]
    params = ["a", "b"]
    rates = adam.get_learning_rate(grads, params)
    assert set(rates.keys()) == set(params)
    # Should be nonzero and opposite sign
    assert rates["a"] > 0
    assert rates["b"] < 0


def test_adam_learning_rate_str():
    """Test string representation of AdamLearningRate."""
    adam = AdamLearningRate()
    s = str(adam)
    assert "AdamLearningRate(" in s


def test_performance_based_learning_rate_increase_and_decrease():
    """Test performance-based learning rate adjustment."""
    perf = PerformanceBasedLearningRate(
        initial_learning_rate=0.1,
        min_learning_rate=0.01,
        max_learning_rate=0.5,
        adjustment_factor=2.0,
    )
    # First call sets previous_performance
    lr1 = perf.get_learning_rate(1.0)
    # Increase
    lr2 = perf.get_learning_rate(2.0)
    assert lr2 > lr1
    # Decrease
    lr3 = perf.get_learning_rate(1.0)
    assert lr3 < lr2
    # Clamp to min
    for _ in range(10):
        perf.get_learning_rate(-100)
    assert perf.learning_rate >= 0.01
    # Clamp to max
    perf = PerformanceBasedLearningRate(
        initial_learning_rate=0.1,
        min_learning_rate=0.01,
        max_learning_rate=0.5,
        adjustment_factor=2.0,
    )
    perf.get_learning_rate(1.0)
    for _ in range(10):
        perf.get_learning_rate(100)
    assert perf.learning_rate <= 0.5


def test_performance_based_learning_rate_str():
    """Test string representation of PerformanceBasedLearningRate."""
    perf = PerformanceBasedLearningRate()
    s = str(perf)
    assert "PerformanceBasedLearningRate(" in s
