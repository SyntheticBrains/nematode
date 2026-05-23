"""*C. elegans* connectome substrate (Phase 6 Tranche 1 / L0).

Public API for downstream consumers. See
``openspec/changes/add-connectome-substrate/`` for the design.
"""

from quantumnematode.connectome.model import (
    CellClass,
    ChemicalSynapse,
    Connectome,
    GapJunction,
    Neuron,
)

__all__ = [
    "CellClass",
    "ChemicalSynapse",
    "Connectome",
    "GapJunction",
    "Neuron",
]
