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
    max_clip_gradient: float | dict = 1.0,
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
        The maximum gradient value for clipping or normalization. If a dict is provided,
        it should map parameter names to their respective maximum gradient values.
        Defaults to 1.0.

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
            # If max_clip_gradient is a dict, use per-parameter clipping
            if isinstance(max_clip_gradient, dict):
                return [
                    max(
                        -max_clip_gradient.get(param, 1.0),
                        min(g, max_clip_gradient.get(param, 1.0)),
                    )
                    for g, param in zip(gradients, max_clip_gradient.keys(), strict=False)
                ]
            return [max(-max_clip_gradient, min(g, max_clip_gradient)) for g in gradients]
