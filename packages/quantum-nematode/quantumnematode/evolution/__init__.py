"""Brain-agnostic evolution framework.

Provides genome encoding, fitness evaluation, lineage tracking, and an
optimisation loop with checkpoint/resume. The framework is brain-agnostic:
any classical brain implementing ``WeightPersistence`` can be plugged in
via a one-class encoder registration.

See ``openspec/specs/evolution-framework/`` for the capability spec.
"""

from quantumnematode.evolution.brain_factory import instantiate_brain_from_sim_config
from quantumnematode.evolution.encoders import (
    ENCODER_REGISTRY,
    NON_GENOME_COMPONENTS,
    GenomeEncoder,
    HyperparameterEncoder,
    LSTMPPOEncoder,
    MLPPPOEncoder,
    build_birth_metadata,
    get_encoder,
    select_encoder,
)
from quantumnematode.evolution.fitness import (
    EpisodicSuccessRate,
    FitnessFunction,
    FrozenEvalRunner,
    LearnedPerformanceFitness,
)
from quantumnematode.evolution.genome import Genome, genome_id_for
from quantumnematode.evolution.inheritance import (
    InheritanceStrategy,
    LamarckianInheritance,
    NoInheritance,
)
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
    "HyperparameterEncoder",
    "InheritanceStrategy",
    "LSTMPPOEncoder",
    "LamarckianInheritance",
    "LearnedPerformanceFitness",
    "LineageTracker",
    "MLPPPOEncoder",
    "NoInheritance",
    "build_birth_metadata",
    "genome_id_for",
    "get_encoder",
    "instantiate_brain_from_sim_config",
    "select_encoder",
]
