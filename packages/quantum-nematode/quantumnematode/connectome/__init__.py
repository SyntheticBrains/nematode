"""*C. elegans* connectome substrate.

Loads the real *C. elegans* 302-neuron connectome (Cook et al. 2019
hermaphrodite) and exposes it through typed Pydantic models for downstream
consumption by brain architectures. Chemical synapses and gap junctions are
represented as separately-typed connections. The same matrices are also
available as released under CC BY 4.0 in Emmons 2024, with the lab's later
corrections, together with the synapses from neurons onto the body wall
muscles.
"""

from quantumnematode.connectome.loader import (
    load_cook_2019_hermaphrodite,
    load_emmons_2024_hermaphrodite,
    load_emmons_2024_neuromuscular,
    load_witvliet_2021_adult,
)
from quantumnematode.connectome.model import (
    CellClass,
    ChemicalSynapse,
    Connectome,
    GapJunction,
    NeuromuscularJunction,
    Neuron,
)
from quantumnematode.connectome.validate import (
    DivergenceReport,
    ValidationResult,
    cross_validate,
    validate_known_pathways,
    validate_neuron_count,
)

__all__: list[str] = [
    "CellClass",
    "ChemicalSynapse",
    "Connectome",
    "DivergenceReport",
    "GapJunction",
    "NeuromuscularJunction",
    "Neuron",
    "ValidationResult",
    "cross_validate",
    "load_cook_2019_hermaphrodite",
    "load_emmons_2024_hermaphrodite",
    "load_emmons_2024_neuromuscular",
    "load_witvliet_2021_adult",
    "validate_known_pathways",
    "validate_neuron_count",
]
