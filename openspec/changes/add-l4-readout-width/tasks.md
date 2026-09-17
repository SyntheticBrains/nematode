# Tasks

## 1. The readout width

- [x] 1.1 A `readout_width: pooled | per_neuron` field on **`ConnectomePPOBrainConfig`**, defaulting
  to `pooled`, plumbed to the topology constructor and passed at the brain's construction site beside
  `enable_gap_junctions` and `synapse_signs`. **No `config_loader` or `dtypes` edit**: config classes
  come from the brain plugin registry and their fields populate generically, which is why neither
  `wiring` nor `perturbation_set` appears in `config_loader`.
- [x] 1.2 **`pooled` must be byte-identical to today.** No extra RNG draw, no reordered draw, no
  changed buffer set. Asserted by test on the parameter shapes and the initial action mean, and
  **verified against data** in 4.2.
- [x] 1.3 **`per_neuron` is initialised by EXPANDING the pooled draw**, not by drawing afresh:
  `W[k, i] = readout[k, class(i)] / |class(i)|`, the classes being unequal (VB 11, DB 7, VA 12, DA 9;
  39 total), so the divisor is per class and not one constant. The `(2, 4)` orthogonal draw still
  happens at the same point with the same shape, so the RNG stream is untouched. A test asserts the
  two widths produce the same action mean on the same input at initialisation **within a stated
  tolerance** — they are equal in exact arithmetic and differ by ~7.45e-9 in float32, because a slice
  `mean()` and a dot product with pre-divided weights round differently. This is the property the
  whole comparison rests on, and it is a statement about the **policy**, not about the runs.
- [x] 1.4 `set_anatomical_readout` expands the same way, so the contrast it writes is the same map at
  both widths. A test asserts it, including that the expanded rows still carry the dorsal/ventral and
  forward/backward contrasts with the per-class `1/|class|` scaling.
- [x] 1.5 The readout's **eligibility** (`pooled_motor` → the 39 raw motor activities) and the
  **symmetric** learning-signal projection both follow the width. `random` routing is unaffected —
  it is what L.0 and R.2 run — but the symmetric branch must not silently keep a `(n, 4)` shape.
- [x] 1.6 **`_N_ACTIONS` currently doubles as the motor-class count and must not be read as the
  readout's input width.** The readout is built as `(readout_out_dim, _N_ACTIONS)` where the second
  dimension is the number of motor **classes**, and `_motor_class_slices` is built with
  `range(_N_ACTIONS)` — both are 4 by coincidence. Where the class count is meant, use
  `len(_MOTOR_CLASSES)`; where the readout's input width is meant, use the width the key selects. A
  test asserts the discrete-action path (whose readout is `(4, 4)`) is unaffected.
- [x] 1.7 Checkpoint identity: a `training_state` written at one width must not load at the other, and
  the task is to **establish which mechanism refuses it** rather than to add a key. The readout is a
  parameter whose shape changes with the width, so the existing shape check is expected to reject the
  load already; `_PLASTICITY_IDENTITY` holds keys whose semantics are *same shapes, different
  meaning*, which is not this case. A test asserts the rejection and names the mechanism that fires.

## 2. The arms

- [x] 2.1 **Four** new configs, each differing from its committed L.0 partner in the `readout_width`
  key **alone** — the two learning arms
  (`..._hard350_eprop_readout_only_wide{,_rewired_null}.yml`) and the two wide floors
  (`..._hard350_eprop_frozen_wide{,_rewired_null}.yml`). Exact-key test, as every arm in this
  programme has had.
- [x] 2.2 The pooled learning arms and the pooled floors are L.0's committed configs, **unchanged**,
  asserted byte-identical to **`b9a5d8f2`**, the commit L.0 ran at.
- [x] 2.3 **Eight arms** × seeds **1–96** *(amended 2026-09-16 after the pilot: 32 first registered, resized below)* — L.0's seeds, because the width-4 cell is L.0's result and
  a paired 2×2 needs the same seeds in every cell. **Four floors, not two**: the widths are the same
  policy but not the same run (1.3), so each learning arm is gated against a floor at its own width.

## 3. Harness

- [x] 3.1 `scripts/analysis/l4_readout_width.py`, the R.1c/R.1d/R.2/L.0 sibling pattern, reusing the
  committed metric and statistics layers verbatim (`t7_continuous_ranking.plateau_tail`,
  `weight_search_architecture_ranking.paired_seed_wilcoxon_bootstrap`, `bh_fdr`).
  **`auc_success` and the censoring counts come from `connectome_structure_efficiency.analyse`,
  called once per width** with this change's wild and rewired arms mapped onto its own two — which
  yields per-seed values for all four cells with that module unmodified. **Do not re-derive AUC
  here**; that duplication is what V.4's review caught in its first draft.
- [x] 3.2 **L.0's harness and block V's two harnesses are READ-ONLY, and a test asserts it.** L.0's
  width-4 numbers are a cell of this 2×2; editing the instrument that produced them would let "the
  instrument changed" compete with the interaction.
- [x] 3.3 **The interaction is the primary**: per seed, `(wt_wide − wt_pooled) − (rn_wide − rn_pooled)`
  on `auc_success`, then the committed paired test. Reported as the primary and named as such.
- [x] 3.4 Both **main effects** reported beside it and never in place of it — a width main effect is
  not evidence about the wiring, and the record must not let it read as one.
- [x] 3.5 **Four learning gates**, each learning arm against its **own** frozen floor, read **before**
  the interaction. An interaction between arms that did not learn is uninterpretable.
- [x] 3.6 **The untrained prior**, wild-type frozen against rewired frozen, measured here rather than
  inherited — and reported as *no pre-update difference detected* rather than as absence, per the
  spec requirement V.4 added.
- [x] 3.7 **`auc_success` is the primary and the reason is carried in the record**, with
  `episodes_to_30pct_success` reported beside it and **its censoring counted per cell of the 2×2**. A
  cell whose censoring rate differs materially from the others voids the censored metric's reading,
  not the primary's.
- [x] 3.8 The five registered readings — `pooling_was_not_the_limit`, `pooling_hid_structure`,
  `width_favours_the_shuffle`, `no_learning`, `insufficient_seeds` — derived from the harness's own
  outputs, with the vocabulary **derived from source rather than hand-copied** (the V.4 lesson).
- [x] 3.9 Tests for 3.1–3.8, including a fixture in each reading, one where a gate fails, and one
  where the two metrics disagree.

## 4. The stop clauses

- [x] 4.1 A **pilot on disjoint seeds 101–104** before any registered seed: all eight arms run, the
  wide arms differ from their pooled partners after training and **not before** (beyond the ~1e-8
  rounding of 1.3), and all four floors sit where a no-learning policy sits. No verdict is read at
  four pairs — the driver withholds it, per V.4.
- [x] 4.2 **Byte-identity check against L.0.** Re-run `wt_pooled` and `rn_pooled` at two of L.0's seeds
  and confirm the logs match L.0's committed runs **exactly**. This is what licenses calling the
  pooled arms the same cell L.0 measured; a mismatch means the `readout_width` key changed the pooled
  path and the 2×2 is not paired.
- [x] 4.3 Re-score L.0's committed `readout_only` arms through **L.0's own harness** and confirm its
  published figures, so the instrument still reproduces the record one cell of this 2×2 comes from.
- [x] 4.4 **The interaction's detectable effect, computed and registered before the campaign runs.**
  *(Amended 2026-09-16, after the pilot and before any registered seed.)* The minimum interaction
  worth detecting is the **sign-flip threshold**: L.0 found the null ahead by **0.1076** on
  `auc_success`, so an interaction must exceed that to mean the pool hid wiring structure. The
  original task assumed the interaction carries √2 the standard error of a single contrast — true
  only if the two widths' wiring differences are **independent**, which this change expected to be
  pessimistic since the widths share a seed's task draws, RNG stream and initial policy. **The pilot
  measured `rho = +0.02`**, realised sd 0.3770 against the 0.4487 independence bound: learning washes
  out the shared start. So the panel is **96 seeds, not 32** — resolving 0.1077, which *matches*
  rather than clears the threshold. Carried as a field, labelled for what it is.
- [x] 4.5 `launch.md` committed before anything runs, carrying the 2×2, the readings, the metric
  choice with its reason, the sensitivity arithmetic and the honest prior.

## 5. Campaign

- [x] 5.1 768 runs: eight arms × seeds **1–96** at 3000 episodes, with `--track-experiment`.
- [x] 5.2 Per-seed CSV with all four cells, the tables and the reading under
  `supporting/066-l4-readout-width/`.

## 6. The record

- [x] 6.1 Logbook 066: the 2×2, the interaction with both main effects, the gates and the prior, both
  metrics with censoring counted, and the sensitivity statement in **every** reading.
- [x] 6.2 The experiments index row and `CHANGELOG.md`.
- [x] 6.3 The tracker's L.1 entry, and the L.4/L.5 gates resolved either way — **reopened** on
  `pooling_hid_structure`, left `closed-unopened` otherwise, with the reason recorded.
- [x] 6.4 *(**does not apply** — the interaction is positive at q = 0.000, so the width objection is answered rather than retired at a sensitivity; recorded as such in Logbook 066.)* **If the interaction is null**: L.0's verdict stands and the width objection is recorded as
  **retired**, with the panel's sensitivity stated so "retired" is not read as "excluded at any size".
- [x] 6.5 **If the interaction is positive**: L.4 and L.5 reopen, and the record states plainly that
  the result says the pool hid *something* and not *what* — V.2 found no graph property predicting
  learning time.
