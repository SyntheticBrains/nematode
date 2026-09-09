# Tasks: structured, pathway-specific instruction

## 1. The instructive pathway

- [ ] 1.1 Derive the instructed neuron set from the classification table and a loaded connectome:
  neurons receiving a chemical edge from a dopaminergic, serotonergic or octopaminergic source.
- [ ] 1.2 Build the per-synapse mask keyed on the post-synaptic neuron, and expose the instructed
  fraction beside it. State at the definition that this models aminergic reach by synaptic
  connectivity and is a lower bound, since transmission here is substantially extrasynaptic.
- [ ] 1.3 Tests: the 12 aminergic neurons are exactly those the atlas marks; the instructed set is
  123 of 302 and the mask selects 2,024 of 3,709 synapses (54.6%); the mask keys on the
  post-synaptic neuron; a rewired connectome derives its own pathway and a different fraction.

## 2. The routed third factor

- [ ] 2.1 Add the routing mode to the plasticity mixin; refuse `pathway` where no pathway exists —
  on the connectome config beside the sign validators, duplicated as a brain-construction guard,
  and outright on the MLP config.
- [ ] 2.2 Apply the modulator only on the instructed set, `1.0` elsewhere, inside the existing
  Hebbian term.
- [ ] 2.3 Register the mask as a wiring buffer so a weight file saved under one routing model is
  refused by the other, as the sign model already is.
- [ ] 2.4 Tests: instructed entries take the global rule's update and uninstructed entries take the
  unmodulated rule's, exactly; routing is a no-op in unmodulated mode; every refusal, including the
  construction guard on a `model_copy`-derived config; the cross-model weight-file refusal.

## 3. Telemetry

- [ ] 3.1 Report the instructed fraction and the instructed share of the update's magnitude,
  measured on the effective update; record both in the shared plasticity report.
- [ ] 3.2 Tests: the share is the whole under `global`; under `pathway` the modulated part of the
  update is confined to the instructed set; both keys are present in the history record.

## 4. Byte-identity and integration

- [ ] 4.1 Test: with `global`, a fixed-seed run's weight trajectory is bit-identical.
- [ ] 4.2 Test: routing composes with the decorrelating terms, consolidation, Dale's law and
  homeostasis in the registered order, with the clamp last.
- [ ] 4.3 Smoke-test entry for one routed config.

## 5. The registered test

- [ ] 5.1 Four arm configs: wild-type and rewired-null plastic arms under each routing mode, each
  one key off its committed parent.
- [ ] 5.2 An analysis harness fixing the four-arm registry, the S1–S4 family, the ordered verdict
  map, the instructed-fraction and instructed-share annotations, and the extension list, with panel
  1's committed global values read as a consistency check and never as a comparator.
- [ ] 5.3 Tests for the harness: each verdict branch including `no_routing_effect`; the committed
  values are not used as a comparator; a missing run is reported and never imputed; S3 and S4
  cannot change the verdict.
- [ ] 5.4 Commit the launch record — protocol, family, verdict map, the extrasynaptic caveat and
  the instructed fraction — then run the four arms (seeds 1–16, 3000 episodes).
- [ ] 5.5 Records under `docs/experiments/logbooks/supporting/047-l4-structured-instruction/`:
  `launch.md`, `panel.json`, `per-seed.csv`, `curves.csv`, `_manifest.txt`, `details.md`.

## 6. Documentation

- [ ] 6.1 `docs/architectures.md`: the routing mode, the pathway's derivation and its status as a
  synaptic proxy for aminergic reach.
- [ ] 6.2 `CHANGELOG.md`.
- [ ] 6.3 Tracker and roadmap updated with the verdict at close-out.
