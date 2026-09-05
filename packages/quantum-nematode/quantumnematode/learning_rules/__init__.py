"""Learning rules operating on brain topologies.

This package hosts ``LearningRule`` implementations — the update mechanisms
paired with ``BrainTopology`` substrates via the seam in
``brain/arch/_rule.py`` / ``_topology.py``. Its first citizen is
``ConnectomePPORule``, the connectome brain's extracted PPO update; beside
it sits ``ThreeFactorRule``, reward-modulated Hebbian plasticity over any
substrate exposing the ``PlasticTopology`` seam (``ConnectomeThreeFactorRule``
is its original name, kept as an alias).

Not to be confused with ``quantumnematode.plasticity``, which is the
quantum-plasticity *evaluation protocol* (sequential multi-objective /
catastrophic-forgetting metrics), not learning rules.
"""

from quantumnematode.learning_rules.ppo import ConnectomePPOBatch, ConnectomePPORule
from quantumnematode.learning_rules.three_factor import (
    ConnectomeThreeFactorRule,
    ThreeFactorBatch,
    ThreeFactorRule,
)

__all__ = [
    "ConnectomePPOBatch",
    "ConnectomePPORule",
    "ConnectomeThreeFactorRule",
    "ThreeFactorBatch",
    "ThreeFactorRule",
]
