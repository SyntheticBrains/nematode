"""What one environment step is in worm time.

A full-speed step moves the worm ``max_step_mm``, so its duration in worm-seconds is that
distance at the crawl speed. The crawl speed and the undulation period are the kinematic figures
the body work is validated against; at a ``max_step_mm`` of one body length (1 mm) a full-speed
step is 5 worm-seconds, about three undulation periods. The figure inherits the crawl speed's
uncertainty: 0.15-0.3 mm/s gives 3.3-6.7 s.
"""

from __future__ import annotations

# Forward crawl speed on agar, in mm/s.
CRAWL_SPEED_MM_PER_S = 0.2
# Period of one undulation while crawling, in seconds.
UNDULATION_PERIOD_S = 1.6


def step_worm_seconds(max_step_mm: float) -> float:
    """Return the worm-time duration, in seconds, of a full-speed step of ``max_step_mm``."""
    if max_step_mm < 0.0:
        msg = f"max_step_mm must be non-negative, got {max_step_mm}"
        raise ValueError(msg)
    return max_step_mm / CRAWL_SPEED_MM_PER_S


def undulations_per_step(max_step_mm: float) -> float:
    """Return how many undulation periods a full-speed step of ``max_step_mm`` spans."""
    return step_worm_seconds(max_step_mm) / UNDULATION_PERIOD_S
