"""Gradient calculation methods for quantum nematode optimizers."""

from enum import Enum


class GradientCalculationMethod(Enum):
    """Enum for gradient calculation methods."""

    RAW = "raw"  # Keep gradients as is
    NORMALIZE = "normalize"  # Normalize gradients
    CLIP = "clip"  # Clip gradients


def compute_gradients(
    gradients: list[float],
    method: GradientCalculationMethod,
    max_clip_gradient: float = 1.0,
) -> list[float]:
    """
    Compute gradients using the specified method.

    Parameters
    ----------
    gradients : list[float]
        The raw gradients to process.
    method : GradientCalculationMethod
        The method to use for processing gradients.
    max_clip_gradient : float, optional
        Maximum allowable gradient norm for clipping, by default 1.0.

    Returns
    -------
    list[float]
        Processed gradients.
    """
    match method:
        case GradientCalculationMethod.RAW:
            return gradients
        case GradientCalculationMethod.NORMALIZE:
            max_gradient = max(abs(g) for g in gradients)
            return [g / max_gradient if max_gradient > 0 else g for g in gradients]
        case GradientCalculationMethod.CLIP:
            return [max(-max_clip_gradient, min(g, max_clip_gradient)) for g in gradients]
