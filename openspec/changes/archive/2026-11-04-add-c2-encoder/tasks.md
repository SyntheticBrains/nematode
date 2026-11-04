## 1. The baseline

- [x] 1.1 Record a connectome identity baseline before any edit: the hand-placed and data-layer arms in the batched evaluator, on all four v3 assays, on discarded seeds (`build/c2-encoder/identity`). The data-layer arm comes from the modulated assay configs with a `modulator_layer: data` override.

## 2. The encoder

- [x] 2.1 `connectome/encoder.py`:
  - the parameter table (names, blocks, ranges, kinds) for `per_couple` and `per_ligand` tying;
  - the unit box, `write_encoder` and `read_encoder`;
  - the class and neuron index maps: amphid classes, CEP/ADE/PDE, transmitter groups, the monoamine mask, ligand groups;
  - the composition, as functions of a parameter tensor that accept a leading candidate axis.
- [x] 2.2 `connectome_ppo.py`:
  - `encoder: compact` and its refusals (strength-normalised additive gaps among them);
  - the fixed wiring (count-normalised signed weights, the scales and group gains, strength-normalised gaps × scale), composed per forward into one recurrent matrix;
  - the class-tied sensory input with antisymmetric lateral, and food contact;
  - the readout bias;
  - the two timescales;
  - the encoder's tensors as the policy's only learnable parameters under PPO, with the critic unchanged;
  - the per-step runner's food-contact feed.
- [x] 2.3 `BatchedConnectome`: the encoded brain, food contact, the live reads. The tax-4 encoding helper. Brain reuse per wiring, rebuilt on a changed wiring or mutant.
- [x] 2.4 Tests for every spec scenario:
  - identity off;
  - refusals;
  - seed-independence of weights;
  - lateral antisymmetry;
  - food-contact targets;
  - round trip;
  - the live write;
  - the timescale scope;
  - tax-4;
  - per-ligand tying (132);
  - a reused brain equal to a fresh one;
  - the encoder and its mutants on every existing null wiring and the receptor-placement null.
- [x] 2.5 Layered parity for the encoded brain on the wild type and on `rewired_degree_preserving` (seeds 5001–5032). The identity check against 1.1: every record byte-identical. The existing parity tests pass.

## 3. Configs

- [x] 3.1 `generate_assay_configs.py`: the `datalayer` arm with the encoder; the four v3 assay configs; `benchmark_grade.ARMS["connectome_c2"]`; a smoke that grades one encoded population.

## 4. The probes (description, discarded seeds)

- [x] 4.1 `scripts/analysis/c2_encoder_probe.py cost|feasibility|readiness`.
- [x] 4.2 The cost pilot:
  - generation pace on D_fit's assays and all four, with brain reuse;
  - a profile of where the time goes, and the byte-identical speed-ups it names;
  - the settling step at 3,200 animals on CPU and on MPS;
  - projections for F.0b-ii, C.2 and W.
- [x] 4.3 The feasibility check: a hand-set vector's post-removal decay constant against 6–12 min.
- [x] 4.4 The readiness probe *(amended 2026-11-03 and 2026-11-04, the maintainer's decisions: the first box stalled, and every one-stage fit from the box centre stalled off the lawn; the box was rescaled twice and a staged fit found, which holds the lawn at a score of 3.43; the restarts below move to the change that registers the staged procedure)*:
  - up to 3 restarts against v3's D_fit (CMA 5980–5982), stopping after 2 pass, with the tightened guard and a cap of 1,000;
  - each answer graded with presence on 5 populations, on all five families and the three mutants;
  - families, bounds, reversal share and timescales reported;
  - the restart-to-restart spread of held-out families and mutant differences against W's minimum;
  - a warm-start existence test if only one passes.
- [x] 4.5 The W sizing record (`supporting/122-c2-encoder/w_sizing.json`): the measured generations per fit, seconds per generation, spread and draws per arm, the compute path, and the tying the evidence favours, for F.0b-ii's and W's registrations. *(Written with what the probes measured: the pace, the compute path, the staged fit's generations and W's projected cost; the spread, the tying and convergence are named as not measured.)*

## 5. Close-out

- [x] 5.1 Logbook 122; the tracker *(the open question, the tying, is not answered: the restarts that would measure it did not run)*:

  - F.0b-ii and C.2;
  - design.md's open question answered;
  - W's inheritance recorded: the gap model, null encoding, compute path and sizing; the class-preserving null still to build.

  AGENTS.md; the CHANGELOG.

- [ ] 5.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
