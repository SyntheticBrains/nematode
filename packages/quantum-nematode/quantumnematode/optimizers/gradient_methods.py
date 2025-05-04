from enum import Enum
import logging

logger = logging.getLogger(__name__)

from enum import Enum
import logging

logger = logging.getLogger(__name__)

class GradientCalculationMethod(Enum):
    """Enum for gradient calculation methods."""
    RAW = "raw"  # Keep gradients as is
    NORMALIZE = "normalize"  # Normalize gradients
    CLIP = "clip"  # Clip gradients

def compute_gradients(
    gradients: list[float],
    method: GradientCalculationMethod,
    max_gradient_norm: float = 1.0,
) -> list[float]:
    """
    Compute gradients using the specified method.

    Parameters
    ----------
    gradients : list[float]
        The raw gradients to process.
    method : GradientCalculationMethod
        The method to use for processing gradients.
    max_gradient_norm : float, optional
        Maximum allowable gradient norm for clipping, by default 1.0.

    Returns
    -------
    list[float]
        Processed gradients.
    """
    if method == GradientCalculationMethod.RAW:
        return gradients

    if method == GradientCalculationMethod.NORMALIZE:
        max_gradient = max(abs(g) for g in gradients)
        normalized_gradients = [g / max_gradient if max_gradient > 0 else g for g in gradients]
        logger.debug(f"Normalized gradients: {normalized_gradients}")
        return normalized_gradients

    if method == GradientCalculationMethod.CLIP:
        clipped_gradients = [
            max(-max_gradient_norm, min(g, max_gradient_norm)) for g in gradients
        ]
        logger.debug(f"Clipped gradients: {clipped_gradients}")
        return clipped_gradients