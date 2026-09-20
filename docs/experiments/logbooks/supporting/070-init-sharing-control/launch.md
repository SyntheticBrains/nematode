# A.1 — the init-vs-rewiring control: registration and launch

**Registered 2026-09-20, before any panel seed ran.** Change: `add-init-sharing-control`.
Roadmap: § Phase 8 **D15**, as corrected at PR #395's review.

Everything below is fixed in advance. The pilot (seeds 105–108, thermal cell, 48 runs) had run when
this was written; **no panel seed had**.

## The question

Block V's wiring advantage — the wild type reaching competence 23–55% sooner than its
degree-preserving rewired null under PPO — carries a standing condition: rewiring and initialisation
vary together. Dhiman 2026 reports that advantage dissolving in the fly under shared initialisation
plus a degree-preserving null.

Reading the initialisation path narrows what "shared" means here. The draw is per-edge over the
`(pre, post)`-sorted edge list; both graphs carry 3,709 edges, so they already consume the same
standard-normal stream and the *n*-th value matches. **What differs is which edge the *n*-th value
lands on.** Everything peripheral — readout, gains, critic — is already byte-identical across
wirings at one seed, asserted by a committed test.

## The design

Wiring {wild type, rewired null} × draw {`edge_order`, `dense_mask`, `per_neuron_fanin`}, both
cells, **32 paired seeds (129–160)**, four arms per cell per mode. **768 runs.**

The **primary is the interaction** of draw mode with wiring, one per sharing mode per cell, paired
by seed: a seed's wiring gap under a shared draw minus the same seed's gap under the committed one.
A draw mode that moves both arms equally is a fact about the draw, not the wiring.

The `edge_order` level is **re-run**, not reused from V.4: reuse requires a parsed-field identity
check with no partial reuse, and the dependency set moved since those runs.

## The metric, decided by a rule fixed before the rates are known

`episodes_to_30pct_success` is right-censored at the horizon and the interaction is a difference of
differences across four cells — the pairing the metric requirement forbids unless censoring is
comparable across them. So censoring is counted **per cell, never pooled**; if the spread exceeds
**0.10** the uncensored `auc_success` carries the interaction and the censored metric is reported
beside it; otherwise the reverse. **Both are always reported.**

*(Pilot observation, not a result: the spread was 0.000 on the thermal cell, so on that evidence the
censored metric is valid here.)*

## Power, and why the panel is 32 seeds rather than D15's floor of 16

Sensitivity is computed from **V.4's committed per-seed spread** (`supporting/065-…/per-seed-primary.csv`,
128 rows) — the frozen prior-committed source the requirement names. Four pilot seeds cannot estimate
a spread.

The interaction's per-seed variance exceeds either single contrast's, by
`sd_interaction = sd_gap × sqrt(2(1−ρ))`. **The pilot measured ρ and found nothing usable**: −0.66 and
+0.34 on the censored metric, −0.33 and +0.17 on the uncensored, at n = 4. So sensitivity is computed
at **ρ = 0**, the conservative case.

Minimum detectable interaction, as a fraction of the effect it would have to cancel:

| cell | metric | n = 16 | **n = 32** |
|---|---|---|---|
| thermal | `episodes_to_30pct_success` | 1.25 | **0.88** |
| thermal | `auc_success` | 0.92 | **0.65** |
| hard_food | `episodes_to_30pct_success` | 1.14 | **0.81** |
| hard_food | `auc_success` | 0.59 | **0.42** |

At 16 the censored metric could not have detected a **total** dissolution on either cell. At 32 both
metrics can, and the uncensored metric reaches the two-thirds bar the ablation precedent uses. A
control that exists to answer a published critique should not be unable to see the answer, so the
panel is 32.

## The registered minimum, in both directions

As a fraction of the **within-campaign** baseline wiring effect, not of block V's published figure:

- **Dissolution** — an interaction removing **≥ 2/3** of the baseline effect, significant at q < 0.05.
- **Survival** — an interaction whose interval excludes a 2/3 reduction.
- **Shrunken** — significant, directionally a reduction, but **below** 2/3: reported as shrunken with
  both the observed size and the minimum stated, never as dissolution and never as survival.
- **The reverse direction** — an interaction that *increases* the wiring effect — carries the **same**
  2/3 minimum. L.1b is why: it guarded one direction and left the other free, and the unguarded
  reading then fired at 47% of the guarded bar.

## Read separately, never pooled

The two cells are read on their own. A split is reported as a split and is evidence about the scope
of block V's generalisation, which it stops being the moment the cells are pooled toward whichever
supports the original claim.

## The branch this campaign's outcome feeds

Registered in the roadmap's risk table before the phase began: if the effect dissolves under both
definitions, that is a **result, not a failure** — the learning-speed claim is restated as an
initialisation effect, and B.1 still runs because measured weights are a different question. If the
`edge_order` arm itself fails to reproduce block V's direction on fresh seeds, **the interaction is
not read at all** and that non-reproduction is the finding.

*(Pilot observation, not a result: at its four seeds the baseline gap ran **opposite** to block V,
−0.108 on `auc_success` against V.4's +0.163. That is well inside one committed standard deviation
of 0.170 and is exactly what four seeds are expected to do. It is recorded because it was seen
before the panel ran, not because it means anything.)*

## Launch

```bash
T=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis
H=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350

uv run python scripts/run_campaign.py \
  --config ${T}_t20.yml --config ${T}_rewired_null_t20.yml \
  --config ${T}_frozen_t20.yml --config ${T}_rewired_null_frozen_t20.yml \
  --config ${T}_t20_densemask.yml --config ${T}_rewired_null_t20_densemask.yml \
  --config ${T}_frozen_t20_densemask.yml --config ${T}_rewired_null_frozen_t20_densemask.yml \
  --config ${T}_t20_fanin.yml --config ${T}_rewired_null_t20_fanin.yml \
  --config ${T}_frozen_t20_fanin.yml --config ${T}_rewired_null_frozen_t20_fanin.yml \
  --config ${H}.yml --config ${H}_rewired_null.yml \
  --config ${H}_frozen.yml --config ${H}_rewired_null_frozen.yml \
  --config ${H}_densemask.yml --config ${H}_rewired_null_densemask.yml \
  --config ${H}_frozen_densemask.yml --config ${H}_rewired_null_frozen_densemask.yml \
  --config ${H}_fanin.yml --config ${H}_rewired_null_fanin.yml \
  --config ${H}_frozen_fanin.yml --config ${H}_rewired_null_frozen_fanin.yml \
  --seeds 129-160 --runs 3000 --output-dir campaigns/init-sharing-control \
  -- --theme headless --track-experiment

uv run python scripts/analysis/init_sharing_control.py \
  --campaign campaigns/init-sharing-control --out-dir build/a1 \
  --out build/a1/init_sharing_control.json
```

Same launch shape as V.4's, which is the point: the instrument and the way it is fed are unchanged.

## Artefact retention (A.0)

Committed: the parsed per-seed CSV, the analysis JSON, this launch record. Archived off-repo: the
raw campaign logs. The campaign directory is removed only after the CSV is committed, and any field
that cannot be compared because its source is gone is named **uncompared** rather than counted as
matching.

## Cost

The pilot ran 48 thermal runs in 2,769 s at 14.8× parallelism, 24 MB. The panel is 16× that in runs,
with half of them on the cheaper `hard350` cell: **roughly 10–13 hours** and under 400 MB.
