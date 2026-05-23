"""*C. elegans* connectome substrate.

Loads the real *C. elegans* 302-neuron connectome (Cook et al. 2019
hermaphrodite) and exposes it through typed Pydantic models for downstream
consumption by brain architectures. Chemical synapses and gap junctions are
represented as separately-typed connections.
"""

from quantumnematode.connectome.loader import (
    load_cook_2019_hermaphrodite,
    load_witvliet_2021_adult,
)
from quantumnematode.connectome.model import (
    CellClass,
    ChemicalSynapse,
    Connectome,
    GapJunction,
    Neuron,
)
from quantumnematode.connectome.validate import (
    DivergenceReport,
    ValidationResult,
    cross_validate,
    validate_known_pathways,
    validate_neuron_count,
)

__all__ = [
    "CellClass",
    "ChemicalSynapse",
    "Connectome",
    "DivergenceReport",
    "GapJunction",
    "Neuron",
    "ValidationResult",
    "cross_validate",
    "load_cook_2019_hermaphrodite",
    "load_witvliet_2021_adult",
    "validate_known_pathways",
    "validate_neuron_count",
]
