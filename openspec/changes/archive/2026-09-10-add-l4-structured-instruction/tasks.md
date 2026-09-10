# Tasks: structured, pathway-specific instruction

## 0. Co-transmitter identities

- [x] 0.1 Extend the atlas reader to every identity column; record each neuron's release
  identities in the classification table beside the primary one sign grounding uses; state the
  exclusion rule (uptake-only; alternative synthesis/uptake; precursor; sex-specific) with the
  excluded names listed at the definition.
- [x] 0.2 Regenerate the table with the generation script; its `--check` mode covers the new
  column.
- [x] 0.3 Tests: ADF and HSN carry serotonin beside acetylcholine and RIM carries tyramine beside
  glutamate; AIM and RIH carry no aminergic release identity; the five qualified entries carry
  none; every grounded sign equals the single-column table's.

## 1. The instructive pathway

- [x] 1.1 Derive the instructed neuron set from the classification table and a loaded connectome:
  neurons receiving a chemical edge from a neuron releasing dopamine, serotonin, octopamine or
  tyramine in any of its identities.
- [x] 1.2 Build the per-synapse mask keyed on the post-synaptic neuron, and expose the instructed
  fraction beside it. State at the definition that this models aminergic reach by synaptic
  connectivity and is a lower bound, since transmission here is substantially extrasynaptic.
- [x] 1.3 Tests: the 18 aminergic neurons are exactly those listed in the design; the instructed set
  is 169 of 302 and the mask selects 2,636 of 3,709 synapses (71.1%); the mask keys on the
  post-synaptic neuron; a rewired connectome derives its own pathway and a different fraction.

## 2. The routed third factor

- [x] 2.1 Add the routing mode to the plasticity mixin; refuse `pathway` where no pathway exists —
  on the connectome config beside the sign validators, duplicated as a brain-construction guard,
  and outright on the MLP config — and refuse it with the unmodulated rule, on the mixin.
- [x] 2.2 Hand the mask to the rule as an aligned per-tensor list at construction, as the sign
  vector is; apply the modulator only on the instructed set, `1.0` elsewhere, inside the existing
  Hebbian term.
- [x] 2.3 Record the routing mode in `training_state` for provenance; a weight file loads across
  routing modes, since routing changes how learning is applied and not what the weights are.
- [x] 2.4 Tests: instructed entries take the global rule's update and uninstructed entries take the
  unmodulated rule's, exactly; every refusal, including the unmodulated-rule one and the
  construction guard on a `model_copy`-derived config; a `global`-saved file loads under
  `pathway` and the clone assay's start points remain loadable.

## 3. Telemetry

- [x] 3.1 Report the instructed fraction and the instructed share of the update's magnitude,
  measured on the effective update; record both in the shared plasticity report.
- [x] 3.2 Tests: the share is the whole under `global`; under `pathway` it is the instructed set's
  share of the effective update's magnitude; both keys are present in the history record.

## 4. Byte-identity and integration

- [x] 4.1 Test: with `global`, a fixed-seed run's weight trajectory is bit-identical.
- [x] 4.2 Test: routing composes with the decorrelating terms, consolidation, Dale's law and
  homeostasis in the registered order, with the clamp last.
- [x] 4.3 Smoke-test entry for one routed config.

## 5. The registered test

- [x] 5.1 Four arm configs: wild-type and rewired-null plastic arms under each routing mode, each
  one key off its committed parent.
- [x] 5.2 An analysis harness fixing the four-arm registry, the S1–S4 family, the ordered verdict
  map, the instructed-fraction and instructed-share annotations, and the extension list, with panel
  1's committed global values read as a consistency check and never as a comparator.
- [x] 5.3 Tests for the harness: each verdict branch including `no_routing_effect`; the committed
  values are not used as a comparator; a missing run is reported and never imputed; S3 and S4
  cannot change the verdict.
- [x] 5.4 Commit the launch record — protocol, family, verdict map, the extrasynaptic caveat and
  the instructed fraction — then run the four arms (seeds 1–16, 3000 episodes).
- [x] 5.5 Records under `docs/experiments/logbooks/supporting/047-l4-structured-instruction/`:
  `launch.md`, `panel.json`, `per-seed.csv`, `curves.csv`, `_manifest.txt`, `details.md`.

## 6. Documentation

- [x] 6.1 `docs/architectures.md`: the routing mode, the pathway's derivation and its status as a
  synaptic proxy for aminergic reach.
- [x] 6.2 `CHANGELOG.md`.
- [x] 6.3 Tracker and roadmap updated with the verdict at close-out.
