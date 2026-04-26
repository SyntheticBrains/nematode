"""Brain-agnostic evolution framework for Phase 5.

Provides genome encoding, fitness evaluation, lineage tracking, and an
optimisation loop with checkpoint/resume. The framework is brain-agnostic:
any classical brain implementing ``WeightPersistence`` can be plugged in
via a one-class encoder registration.

See ``openspec/changes/2026-04-28-add-evolution-framework/`` for the full
design and rationale.
"""

from quantumnematode.evolution.genome import Genome, genome_id_for

__all__ = ["Genome", "genome_id_for"]
