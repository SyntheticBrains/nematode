"""Learning rules operating on brain topologies.

This package hosts ``LearningRule`` implementations — the update mechanisms
paired with ``BrainTopology`` substrates via the seam in
``brain/arch/_rule.py`` / ``_topology.py``. Its first citizen is
``ConnectomePPORule``, the connectome brain's extracted PPO update; future
rules (e.g. three-factor plasticity) land beside it.

Not to be confused with ``quantumnematode.plasticity``, which is the
quantum-plasticity *evaluation protocol* (sequential multi-objective /
catastrophic-forgetting metrics), not learning rules.
"""

from quantumnematode.learning_rules.ppo import ConnectomePPOBatch, ConnectomePPORule

__all__ = ["ConnectomePPOBatch", "ConnectomePPORule"]
