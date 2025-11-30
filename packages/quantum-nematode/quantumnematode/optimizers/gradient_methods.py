"""Gradient calculation methods for quantum nematode optimizers."""

from enum import Enum

import numpy as np

DEFAULT_MAX_CLIP_GRADIENT = 1.0
DEFAULT_MAX_GRADIENT_NORM = 0.5


class GradientCalculationMethod(Enum):
    """Gradient calculation methods."""

    RAW = "raw"  # Keep gradients as is
    NORMALIZE = "normalize"  # Normalize gradients
    CLIP = "clip"  # Clip gradients
    NORM_CLIP = "norm_clip"  # Clip gradient vector by norm


def compute_gradients(
    gradients: list[float],
    method: GradientCalculationMethod,
    max_clip_gradient: float | dict = DEFAULT_MAX_CLIP_GRADIENT,
    max_gradient_norm: float = DEFAULT_MAX_GRADIENT_NORM,
) -> list[float]:
    """
    Compute gradients using the specified method.

    Parameters
    ----------
    gradients : list[float]
        The raw gradients to process.
    method : GradientCalculationMethod
        The method to use for processing gradients.
    max_clip_gradient : float | dict, optional
        The maximum gradient value for clipping or normalization.
        If a dict is provided, it should map parameter names to their respective
        maximum gradient values (used with NORMALIZE and CLIP methods).
        Defaults to 1.0.
    max_gradient_norm : float, optional
        The maximum norm for gradient vector clipping (used with NORM_CLIP method).
        Defaults to 0.5.

    Returns
    -------
    list[float]
        Processed gradients.
    """
    match method:
        case GradientCalculationMethod.RAW:
            return gradients
        case GradientCalculationMethod.NORMALIZE:
            # If max_clip_gradient is a dict, use per-parameter normalization
            if isinstance(max_clip_gradient, dict):
                return [
                    g / max(abs(g), abs(max_clip_gradient.get(param, 1.0)))
                    if max_clip_gradient.get(param, 1.0) > 0
                    else g
                    for g, param in zip(gradients, max_clip_gradient.keys(), strict=False)
                ]
            max_gradient = max(abs(g) for g in gradients)
            return [g / max_gradient if max_gradient > 0 else g for g in gradients]
        case GradientCalculationMethod.CLIP:
            # If max_clip_gradient is a dict, set clip to 1.0 for all gradients
            if isinstance(max_clip_gradient, dict):
                max_clip_gradient = 1.0

            return [max(-max_clip_gradient, min(g, max_clip_gradient)) for g in gradients]
        case GradientCalculationMethod.NORM_CLIP:
            # Clip gradient vector by total norm, based on PyTorch's clip_grad_norm_
            grad_norm = np.linalg.norm(gradients)
            if grad_norm > max_gradient_norm:
                scale = max_gradient_norm / grad_norm
                return [float(g * scale) for g in gradients]
            return list(gradients)
