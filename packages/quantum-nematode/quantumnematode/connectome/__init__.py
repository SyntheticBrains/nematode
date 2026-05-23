"""*C. elegans* connectome substrate.

Loads the real *C. elegans* 302-neuron connectome (Cook et al. 2019
hermaphrodite) and exposes it through typed Pydantic models for downstream
consumption by brain architectures. Chemical synapses and gap junctions are
represented as separately-typed connections.
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
