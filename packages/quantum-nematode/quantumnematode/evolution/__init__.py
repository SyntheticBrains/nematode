"""Brain-agnostic evolution framework for Phase 5.

Provides genome encoding, fitness evaluation, lineage tracking, and an
optimisation loop with checkpoint/resume. The framework is brain-agnostic:
any classical brain implementing ``WeightPersistence`` can be plugged in
via a one-class encoder registration.

See ``openspec/changes/2026-04-28-add-evolution-framework/`` for the full
design and rationale.
"""

from quantumnematode.evolution.brain_factory import instantiate_brain_from_sim_config
from quantumnematode.evolution.encoders import (
    ENCODER_REGISTRY,
    NON_GENOME_COMPONENTS,
    GenomeEncoder,
    LSTMPPOEncoder,
    MLPPPOEncoder,
    get_encoder,
)
from quantumnematode.evolution.fitness import (
    EpisodicSuccessRate,
    FitnessFunction,
    FrozenEvalRunner,
)
from quantumnematode.evolution.genome import Genome, genome_id_for
from quantumnematode.evolution.lineage import CSV_HEADER, LineageTracker
from quantumnematode.evolution.loop import CHECKPOINT_VERSION, EvolutionLoop

__all__ = [
    "CHECKPOINT_VERSION",
    "CSV_HEADER",
    "ENCODER_REGISTRY",
    "NON_GENOME_COMPONENTS",
    "EpisodicSuccessRate",
    "EvolutionLoop",
    "FitnessFunction",
    "FrozenEvalRunner",
    "Genome",
    "GenomeEncoder",
    "LSTMPPOEncoder",
    "LineageTracker",
    "MLPPPOEncoder",
    "genome_id_for",
    "get_encoder",
    "instantiate_brain_from_sim_config",
]
