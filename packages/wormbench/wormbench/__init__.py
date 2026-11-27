"""Grade simulated or real *C. elegans* tracks against real-worm behaviour.

The benchmark reads 1-s positions alone and grades them on standard on-lawn and off-food assays:
Scheer & Bargmann's small lawn and uniform well, López-Cruz et al.'s local search after food
removal, the kinematics and activity of worms on a lawn, and Chen et al.'s butanone chemotaxis.
Each statistic is graded against bands built from the spread between real recording dates.

The package depends on numpy and scipy only, and imports nothing outside itself.
"""

from __future__ import annotations

__version__ = "1.0.0"
