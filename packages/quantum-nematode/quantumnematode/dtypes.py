"""Core type definitions for Quantum Nematode.

This module provides type aliases used across multiple submodules
for representing positions, paths, and trajectory data.
"""

# =============================================================================
# Position Types
# =============================================================================

# Grid position (discrete coordinates used in environment)
GridPosition = tuple[int, int]

# Continuous position (used in calculations like chemotaxis)
Position = tuple[float, float]

# =============================================================================
# Gradient Types
# =============================================================================

# Gradient in polar coordinates (magnitude, direction_radians)
# Used by temperature, food, and predator gradient computations
GradientPolar = tuple[float, float]

# Gradient in Cartesian coordinates (dx, dy)
GradientVector = tuple[float, float]

# =============================================================================
# Spot Types (for temperature field)
# =============================================================================

# Hot/cold spot definition: (x, y, intensity)
# Intensity is temperature delta in °C (positive for hot, used as-is for cold)
TemperatureSpot = tuple[int, int, float]

# =============================================================================
# Path and History Types
# =============================================================================

# Agent's path through the environment (sequence of grid positions)
AgentPath = list[GridPosition]

# Food positions at a single timestep
FoodPositions = list[GridPosition]

# Food positions over time (one entry per timestep)
FoodHistory = list[FoodPositions]

# Continuous versions for calculations
PositionPath = list[Position]
PositionFoodHistory = list[list[Position]]
