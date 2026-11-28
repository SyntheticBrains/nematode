# Investigation records

This directory holds two pieces of side work from Phase 10. Both are closed. They are kept as a
record of what was asked and what ran, not as live code: nothing here is imported, tested or run by
the project, and the Python files are kept as `.py.txt`.

- **The fit-first investigation** (closed 2026-11-14). It asked whether fitting a connectome model
  to neural recordings would reach the project's goal faster than behaviour fitting. Its brief is
  `BRIEF.md`, its conclusion `RECOMMENDATION.md`, and its handover to the Phase 10 work
  `HANDOVER_PHASE10.md`. The readings are [Logbook 134](../docs/experiments/logbooks/134-fit-first-neural-readings.md):
  unfitted, C.2's encoder on the true wiring predicts Creamer et al.'s measured responses to
  single-neuron stimulation better than all 19 degree-preserving rewirings. Behaviour fitting's part
  is unsettled: at Logbook 131's answer it does not detectably add, and at C.2's registered answer
  it raised r by 0.023, on one answer, as description (Logbook 138). The spike's code is in `spike/`.
- **The neural-readout ceiling spike** (closed 2026-11-16). It measured how far that readout can
  reach. [Logbook 135](../docs/experiments/logbooks/135-neural-readout-ceiling.md) records it: the
  ceiling is about 0.42, and a linear model learned on the anatomical support does not beat the
  calibrated unfitted network under it. The code is in `neural-ceiling/` (see its README).

What these readings mean for the next phase is weighed in
[Logbook 138](../docs/experiments/logbooks/138-phase10-synthesis.md), § The shipment decision.
