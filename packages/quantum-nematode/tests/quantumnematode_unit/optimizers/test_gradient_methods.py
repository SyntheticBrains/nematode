from quantumnematode.optimizers.gradient_methods import (
    GradientCalculationMethod,
    compute_gradients,
)

def test_raw_method_returns_gradients_unchanged():
    """Test that the RAW method returns gradients unchanged."""
    grads = [1.0, -2.0, 0.5]
    result = compute_gradients(grads, GradientCalculationMethod.RAW)
    assert result == grads

def test_normalize_method_scales_by_max_abs():
    """Test that the NORMALIZE method scales gradients by the maximum absolute value."""
    grads = [2.0, -4.0, 1.0]
    result = compute_gradients(grads, GradientCalculationMethod.NORMALIZE)
    max_grad = max(abs(g) for g in grads)
    expected = [g / max_grad for g in grads]
    assert result == expected

def test_normalize_method_zero_max():
    """Test that the NORMALIZE method returns gradients unchanged when max is zero."""
    grads = [0.0, 0.0, 0.0]
    result = compute_gradients(grads, GradientCalculationMethod.NORMALIZE)
    assert result == grads

def test_clip_method_clips_to_max():
    """Test that the CLIP method clips gradients to the specified maximum value."""
    grads = [2.0, -3.0, 0.5]
    max_clip = 1.0
    result = compute_gradients(grads, GradientCalculationMethod.CLIP, max_clip)
    expected = [1.0, -1.0, 0.5]
    assert result == expected

def test_clip_method_with_default():
    """Test that the CLIP method uses the default max value when not specified."""
    grads = [2.0, -3.0, 0.5]
    result = compute_gradients(grads, GradientCalculationMethod.CLIP)
    expected = [1.0, -1.0, 0.5]
    assert result == expected

def test_normalize_method_with_dict():
    """Test that the NORMALIZE method works with a dictionary of max values."""
    grads = [2.0, -4.0]
    max_clip_gradient = {"a": 2.0, "b": 4.0}
    result = compute_gradients(grads, GradientCalculationMethod.NORMALIZE, max_clip_gradient)
    expected = [2.0 / 2.0, -4.0 / 4.0]
    assert result == expected

def test_normalize_method_with_dict_zero_clip():
    """Test that the NORMALIZE method returns gradients unchanged when max is zero in dict."""
    grads = [2.0, -4.0]
    max_clip_gradient = {"a": 0.0, "b": 0.0}
    result = compute_gradients(grads, GradientCalculationMethod.NORMALIZE, max_clip_gradient)
    assert result == grads

def test_clip_method_with_dict_forces_clip_to_1():
    """Test that the CLIP method clips gradients to 1.0 when using a dict."""
    grads = [2.0, -3.0, 0.5]
    max_clip_gradient = {"a": 2.0, "b": 4.0, "c": 1.0}
    result = compute_gradients(grads, GradientCalculationMethod.CLIP, max_clip_gradient)
    expected = [1.0, -1.0, 0.5]
    assert result == expected
