# Tasks: Vendor the measured synaptic weights, and a `weight_prior` switch

Phase 8 task **B.1a**. The plan is authoritative in `docs/roadmap.md` § Phase 8 **D17**; the decisions
taken before implementation are in this change's `design.md`. No campaign runs in this change.

- [x] 1. **Vendor the table** — **done: the pinned bytes, SHA256 matching. The upstream file uses CRLF, which `* text=auto` would have normalised on commit and broken the digest, so it is marked `-text` in `.gitattributes`.** Original scope: — `data/connectome/creamer_lds_2026_model_weights.csv`, the upstream
  `quick_start_examples/model_weights.csv` at commit `bba43302d50a4947804d98b01779856e648237cc`
  byte-for-byte (SHA256 `f452b88461aa90d414fd652e246e11302293d207510b9f4c94bf9a9a8098924c`,
  41,959 bytes), with `LICENSE-creamer-lds.txt` carrying the MIT notice. Named after its source
  repository, `Creamer_LDS_2026`, because the preprint (2024), its v3 (2025) and the repository (2026)
  each carry a different year.

- [x] 2. **Provenance** — **done, including the atlas's and this file's lines in `## Verification`.** Original scope: — a `PROVENANCE.md` entry in the house shape: description, upstream path and
  commit, size, SHA256, source URL, licence, retrieval date, paper (bioRxiv 2024.09.22.614271 v3,
  a preprint), redistribution rationale. A "What is NOT vendored" entry for the Randi 2023 atlas
  (OSF `e2syt`: no licence; `funatlas.h5`: GPL-3.0), Creamer's model pickles and both OSF deposits,
  with the reason for each. Add the transmitter atlas's missing SHA line to `## Verification`.

- [x] 3. **Loader** — **done: `measured_weights.py`, reproducing every planned coverage figure.** Original scope: — `quantumnematode/connectome/measured_weights.py`: read with the SHA256 checked,
  names validated against the canonical classification, a frozen `(pre, post) → weight` table, and a
  coverage report against a connectome (covered; head scope with its self-loops apart; gap-junction
  only, treating gap junctions as undirected; no connection).

- [x] 4. **Loader tests** — **done, 10 tests.** Original scope: — `tests/.../connectome/test_measured_weights.py`: file present and digest
  matches; a changed file refused; every name known; coverage pinned on Cook 2019 — 1,049 covered of
  3,709; 1,386 at head scope, 23 of them self-loops, so 1,363 coverable; 635 positive and 414
  negative; 265 on a gap junction only; 697 on no connection; none onto the 39 body motor neurons.

- [x] 5. **The prior on the brain** — **done. **One correction to the approved design**: normalising the fitted values by their own RMS left the covered edges 1.275x the draw's expected magnitude — deterministic, because the large values sit on neurons with few inputs and a large per-neuron scale. The constant now has the per-neuron scale inside it; spec, design and proposal updated with the reason.** Original scope: — `connectome_ppo.py`: `weight_prior` and
  `measured_weight_scale` beside `weight_draw`; RMS-normalised, per-neuron-scaled values in the edge
  loop after the unchanged `rng.normal` call; the shuffle from the draw generator. **The wild-type
  per-neuron values are computed from the unrewired connectome before rewiring** — the brain rewires
  at `connectome_ppo.py:2217-2222`, before the topology is built — and handed to the topology. On the
  null, each neuron's wild-type values, in their own wild-type pre-index order, land on its first *k*
  incoming edges in pre-index order, for all three measured priors. Refusals, in the validator and in
  `_reject_unsupported_plasticity_modes`: the three untested pairings, and a non-default
  `measured_weight_scale` under `random` or `measured_signs`. `training_state` records the prior and
  multiplier and backfills `weight_draw`. No planning references in package code.

- [x] 6. **Brain tests** — **done, 33 tests, including one pinning the corrected magnitude match.** Original scope: — `tests/.../brain/arch/test_connectome_weight_prior.py`, modelled on
  `test_connectome_weight_draw.py`: default bit-identical; only the chemical weights move under each
  prior (reusing that file's helper over every other parameter); uncovered edges equal the random
  build; covered edges carry the normalised value; `measured_signs` keeps the draw's magnitude;
  `measured_shuffled` has the same multiset in a different placement; on the rewired null, under each
  of the three measured priors, every neuron's first *k* edges carry its wild-type values in wild-type
  pre-index order and the rest equal the random build; a rewired and a wild-type brain at one seed
  agree on the wild-type values; the shared generator ends where it started under every prior; the
  multiplier scales covered edges linearly; every refusal refused at validation and at construction.
  Plus a persistence check that `training_state` carries the new fields.

- [x] 7. **Docs** — **done.** Original scope: — `docs/architectures.md`'s `connectomeppo` row names `weight_prior`; a CHANGELOG
  line; stale docstrings fixed in passing at `connectome/rewiring.py` (says weights "do not affect
  training", false under count-scaled initialisation) and `connectome/neurons.py` (says the
  transmitter column is empty for every neuron).

- [x] 8. **Tracker and roadmap** — **done: B.1a ticked; dated corrections at the tracker, the roadmap's B.1 deliverable and D17.** Original scope: — tick B.1a in `openspec/changes/phase8-tracking/tasks.md`, with a
  dated correction there and at roadmap § B.1 that Randi is cited rather than vendored, and that the
  released table covers 125 neurons (the fitted model 154), not 156.

- [ ] 9. **Close-out** — the full suite via `uv run pytest -m "not nightly"`; `git add -A` then
  `uv run pre-commit run --all-files`, judged by its exit code; a smoke construction under every
  prior on both wirings, and one short run under `measured` from a **temporary config in the
  scratchpad** — B.1a commits no configs, and `run_simulation.py` has no override; archive and PR.
