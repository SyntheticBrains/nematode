"""Validation: chemotaxis metrics, reference values, and behavioural and kinematic instruments.

``adaptive_sensor`` loads the agent and its environment, so its names are imported on first use:
importing a light submodule such as ``roaming_dwelling`` then pulls in none of the simulator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantumnematode.dtypes import (
    Position,
    PositionFoodHistory,
    PositionPath,
)

from .chemotaxis import (
    ChemotaxisMetrics,
    ValidationLevel,
    calculate_chemotaxis_index,
    calculate_chemotaxis_index_stepwise,
    calculate_chemotaxis_metrics,
    calculate_chemotaxis_metrics_stepwise,
)
from .datasets import (
    ChemotaxisDataset,
    LiteratureSource,
    load_chemotaxis_dataset,
)

if TYPE_CHECKING:
    from .adaptive_sensor import (
        StepInputResult,
        run_step_input,
        weber_invariance_peaks,
        weber_spread,
    )

_LAZY = {"StepInputResult", "run_step_input", "weber_invariance_peaks", "weber_spread"}


def __getattr__(name: str) -> Any:  # noqa: ANN401 - any of adaptive_sensor's names
    """Import ``adaptive_sensor``'s names on first access."""
    if name in _LAZY:
        from . import adaptive_sensor

        return getattr(adaptive_sensor, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "ChemotaxisDataset",
    "ChemotaxisMetrics",
    "LiteratureSource",
    "Position",
    "PositionFoodHistory",
    "PositionPath",
    "StepInputResult",
    "ValidationLevel",
    "calculate_chemotaxis_index",
    "calculate_chemotaxis_index_stepwise",
    "calculate_chemotaxis_metrics",
    "calculate_chemotaxis_metrics_stepwise",
    "load_chemotaxis_dataset",
    "run_step_input",
    "weber_invariance_peaks",
    "weber_spread",
]
