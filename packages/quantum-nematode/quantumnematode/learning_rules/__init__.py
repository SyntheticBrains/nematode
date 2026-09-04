"""Learning rules operating on brain topologies (Phase 7 L4, roadmap D8).

This package hosts ``LearningRule`` implementations — the update mechanisms
paired with ``BrainTopology`` substrates via the seam in
``brain/arch/_rule.py`` / ``_topology.py``. Its first citizen is
``ConnectomePPORule``, the connectome brain's extracted PPO update; the
Phase 7 three-factor rules land beside it.

Not to be confused with ``quantumnematode.plasticity``, which is the
quantum-plasticity *evaluation protocol* (sequential multi-objective /
catastrophic-forgetting metrics), not learning rules.
"""

from quantumnematode.learning_rules.ppo import ConnectomePPOBatch, ConnectomePPORule

__all__ = ["ConnectomePPOBatch", "ConnectomePPORule"]
