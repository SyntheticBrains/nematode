# Tasks: Vendor the measured synaptic weights, and a `weight_prior` switch

Phase 8 task **B.1a**. The plan is authoritative in `docs/roadmap.md` § Phase 8 **D17**; the decisions
taken before implementation are in this change's `design.md`. No campaign runs in this change.

- [ ] 1. **Vendor the table** — `data/connectome/creamer_2025_lds_weights.csv`, the upstream
  `quick_start_examples/model_weights.csv` at commit `bba43302d50a4947804d98b01779856e648237cc`
  byte-for-byte (SHA256 `f452b88461aa90d414fd652e246e11302293d207510b9f4c94bf9a9a8098924c`,
  41,959 bytes), with `LICENSE-creamer-lds.txt` carrying the MIT notice.

- [ ] 2. **Provenance** — a `PROVENANCE.md` entry in the house shape: description, upstream path and
  commit, size, SHA256, source URL, licence, retrieval date, paper (bioRxiv 2024.09.22.614271 v3,
  a preprint), redistribution rationale. A "What is NOT vendored" entry for the Randi 2023 atlas
  (OSF `e2syt`: no licence; `funatlas.h5`: GPL-3.0), Creamer's model pickles and both OSF deposits,
  with the reason for each. Add the transmitter atlas's missing SHA line to `## Verification`.

- [ ] 3. **Loader** — `quantumnematode/connectome/measured_weights.py`: read with the SHA256 checked,
  names validated against the canonical classification, a frozen `(pre, post) → weight` table, and a
  coverage report against a connectome (covered, head scope, gap-junction only, no connection).

- [ ] 4. **Loader tests** — `tests/.../connectome/test_measured_weights.py`: file present and digest
  matches; a changed file refused; every name known; coverage pinned on Cook 2019 — 1,049 covered of
  3,709, 1,386 at head scope, 635 positive and 414 negative, 265 on a gap junction only, 697 on no
  connection, none onto the 39 body motor neurons.

- [ ] 5. **The prior on the brain** — `connectome_ppo.py`: `weight_prior` and
  `measured_weight_scale` beside `weight_draw`; RMS-normalised, per-neuron-scaled values in the edge
  loop after the unchanged `rng.normal` call; the shuffle from the draw generator; the rewired null's
  per-neuron multiset in pre-synaptic-index order; the three refused pairings in the validator and in
  `_reject_unsupported_plasticity_modes`; `training_state` records the prior and multiplier and
  backfills `weight_draw`. No planning references in package code.

- [ ] 6. **Brain tests** — `tests/.../brain/arch/test_connectome_weight_prior.py`, modelled on
  `test_connectome_weight_draw.py`: default bit-identical; only the chemical weights move under each
  prior (reusing that file's helper over every other parameter); uncovered edges equal the random
  build; covered edges carry the normalised value; `measured_signs` keeps the draw's magnitude;
  `measured_shuffled` has the same multiset in a different placement; the rewired null's per-neuron
  multisets; the shared generator ends where it started under every prior; the multiplier scales
  covered edges linearly; every refused pairing refused at validation and at construction. Plus a
  persistence check that `training_state` carries the new fields.

- [ ] 7. **Docs** — `docs/architectures.md`'s `connectomeppo` row names `weight_prior`; a CHANGELOG
  line; stale docstrings fixed in passing at `connectome/rewiring.py` (says weights "do not affect
  training", false under count-scaled initialisation) and `connectome/neurons.py` (says the
  transmitter column is empty for every neuron).

- [ ] 8. **Tracker and roadmap** — tick B.1a in `openspec/changes/phase8-tracking/tasks.md`, with a
  dated correction there and at roadmap § B.1 that Randi is cited rather than vendored, and that the
  released table covers 125 neurons (the fitted model 154), not 156.

- [ ] 9. **Close-out** — the full suite via `uv run pytest -m "not nightly"`; `git add -A` then
  `uv run pre-commit run --all-files`, judged by its exit code; a smoke construction under every
  prior on both wirings, and one short run under `measured`; archive and PR.
