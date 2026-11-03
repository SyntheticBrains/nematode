## 1. The baseline

- [ ] 1.1 Record a connectome identity baseline before any edit: the hand-placed and data-layer arms in the batched evaluator, on all four v3 assays, on discarded seeds (`build/c2-encoder/identity`).

## 2. The encoder

- [ ] 2.1 `connectome/encoder.py`: the parameter table (names, blocks, ranges, kinds), the unit box, `write_encoder`, `read_encoder`, and the class and neuron index maps (amphid classes, CEP/ADE/PDE, transmitter groups, monoamine mask).
- [ ] 2.2 `connectome_ppo.py`:
  - `encoder: compact` and its refusals;
  - the fixed wiring (count-normalised signed weights, the scales and group gains, gap × scale, derived inverses rebuilt);
  - the class-tied sensory input with antisymmetric lateral, and food contact;
  - the readout bias;
  - the two timescales;
  - the encoder's tensors as the only learnable parameters under PPO;
  - the per-step runner's food-contact feed.
- [ ] 2.3 `BatchedConnectome`: the encoded brain, food contact, the live reads. The tax-4 encoding helper.
- [ ] 2.4 Tests for every spec scenario:
  - identity off;
  - refusals;
  - seed-independence of weights;
  - lateral antisymmetry;
  - food-contact targets;
  - round trip;
  - the live write;
  - the timescale scope;
  - tax-4.
- [ ] 2.5 Layered parity for the encoded brain (seeds 5001–5032). The identity check against 1.1: every record byte-identical. The existing parity tests pass.

## 3. Configs

- [ ] 3.1 `generate_assay_configs.py`: the `datalayer` arm with the encoder; the four v3 assay configs; `benchmark_grade.ARMS["connectome_c2"]`; a smoke that grades one encoded population.

## 4. The probes (description, discarded seeds)

- [ ] 4.1 `scripts/analysis/c2_encoder_probe.py cost|feasibility|readiness`.
- [ ] 4.2 The cost pilot: generation pace on D_fit's assays and all four; projections for F.0b-ii, C.2 and W.
- [ ] 4.3 The feasibility check: a hand-set vector's post-removal decay constant against 6–12 min.
- [ ] 4.4 The readiness probe: one restart against v3's D_fit, with the tightened guard and a cap of 1,000; the answer graded with presence on 5 populations; families, bounds, reversal share and timescales reported.

## 5. Close-out

- [ ] 5.1 Logbook 122; the tracker (F.0b-ii, C.2; design.md's open question answered); AGENTS.md; the CHANGELOG.
- [ ] 5.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
