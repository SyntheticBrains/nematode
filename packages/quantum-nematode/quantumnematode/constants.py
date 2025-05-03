"""Constants."""

import os

# Feature flags
# NOTE: Pausing is not implemented properly yet.
TOGGLE_PAUSE = os.getenv("TOGGLE_PAUSE", "False").lower() == "false"

# Defaults
DEFAULT_AGENT_BODY_LENGTH = 2
DEFAULT_BRAIN = "simple"
DEFAULT_MAX_STEPS = 100
DEFAULT_MAZE_GRID_SIZE = 5
DEFAULT_SHOTS = 1024
DEFAULT_QUBITS = 2

# Validation
MIN_GRID_SIZE = 5
