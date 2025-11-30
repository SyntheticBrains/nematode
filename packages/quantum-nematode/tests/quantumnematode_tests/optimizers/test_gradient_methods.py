import numpy as np
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


def test_norm_clip_method_clips_by_norm():
    """Test that the NORM_CLIP method clips gradients by the L2 norm."""
    grads = [3.0, 4.0]  # L2 norm = 5.0
    max_norm = 2.5
    result = compute_gradients(
        grads,
        GradientCalculationMethod.NORM_CLIP,
        max_gradient_norm=max_norm,
    )
    # Expected: scale = 2.5 / 5.0 = 0.5, so [1.5, 2.0]
    expected = [1.5, 2.0]
    assert np.allclose(result, expected)


def test_norm_clip_method_no_clipping_when_below_norm():
    """Test that the NORM_CLIP method doesn't clip when norm is below threshold."""
    grads = [0.3, 0.4]  # L2 norm = 0.5
    max_norm = 1.0
    result = compute_gradients(
        grads,
        GradientCalculationMethod.NORM_CLIP,
        max_gradient_norm=max_norm,
    )
    # No clipping should occur
    assert result == grads


def test_norm_clip_method_with_default_norm():
    """Test that the NORM_CLIP method uses the default max norm when not specified."""
    grads = [0.3, 0.4]  # L2 norm = 0.5
    result = compute_gradients(grads, GradientCalculationMethod.NORM_CLIP)
    # Default max_norm = 0.5, so exactly at threshold (no clipping)
    assert result == grads


def test_norm_clip_method_with_zero_gradients():
    """Test that the NORM_CLIP method handles zero gradients correctly."""
    grads = [0.0, 0.0, 0.0]
    result = compute_gradients(
        grads,
        GradientCalculationMethod.NORM_CLIP,
        max_gradient_norm=0.5,
    )
    assert result == grads


def test_norm_clip_method_large_gradient_vector():
    """Test that the NORM_CLIP method scales large gradient vectors correctly."""
    grads = [6.0, 8.0]  # L2 norm = 10.0
    max_norm = 0.5
    result = compute_gradients(
        grads,
        GradientCalculationMethod.NORM_CLIP,
        max_gradient_norm=max_norm,
    )
    # Expected: scale = 0.5 / 10.0 = 0.05, so [0.3, 0.4]
    expected = [0.3, 0.4]
    assert np.allclose(result, expected)


def test_norm_clip_method_with_negative_gradients():
    """Test that the NORM_CLIP method handles negative gradients correctly."""
    grads = [-3.0, -4.0]  # L2 norm = 5.0
    max_norm = 2.5
    result = compute_gradients(
        grads,
        GradientCalculationMethod.NORM_CLIP,
        max_gradient_norm=max_norm,
    )
    # Expected: scale = 2.5 / 5.0 = 0.5, so [-1.5, -2.0]
    expected = [-1.5, -2.0]
    assert np.allclose(result, expected)


def test_norm_clip_method_with_mixed_sign_gradients():
    """Test that the NORM_CLIP method preserves direction with mixed signs."""
    grads = [3.0, -4.0]  # L2 norm = 5.0
    max_norm = 1.0
    result = compute_gradients(
        grads,
        GradientCalculationMethod.NORM_CLIP,
        max_gradient_norm=max_norm,
    )
    # Expected: scale = 1.0 / 5.0 = 0.2, so [0.6, -0.8]
    expected = [0.6, -0.8]
    assert np.allclose(result, expected)


def test_norm_clip_method_preserves_gradient_direction():
    """Test that the NORM_CLIP method preserves the direction of the gradient vector."""
    grads = [1.0, -2.0, 3.0, -4.0]  # L2 norm = sqrt(30) ≈ 5.477
    max_norm = 1.0
    result = compute_gradients(
        grads,
        GradientCalculationMethod.NORM_CLIP,
        max_gradient_norm=max_norm,
    )

    # Verify the result has the correct norm
    result_norm = np.linalg.norm(result)
    assert np.isclose(result_norm, max_norm)

    # Verify direction is preserved (angle between original and result should be 0)
    # Direction is preserved if result = scale * grads for some positive scale
    scale = result_norm / np.linalg.norm(grads)
    expected = [g * scale for g in grads]
    assert np.allclose(result, expected)
