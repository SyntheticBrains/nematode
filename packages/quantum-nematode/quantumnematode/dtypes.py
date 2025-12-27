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
