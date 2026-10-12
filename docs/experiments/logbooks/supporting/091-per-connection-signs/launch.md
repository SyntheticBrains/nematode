# M.10 — the per-connection signs rung: registration and launch

**Registered 2026-10-12, before any panel run.** Change: `add-per-connection-signs-rung`. Both pilots,
the scale calibration and the gate preflight are committed beside this record.

## The question

**Does the wild type's chemical wiring matter once synapse signs are real?** Asked on the thermal
cell at target 35 under PPO, against the chemical-only null.

At random signs the wild type's chemical lead on this cell is +0.016 (Logbook 078). The cell's large
wiring effect, +0.0865, is gap-junction placement, which signs do not touch. The rung asks whether
grounding the signs changes that:

- **per-neuron atlas signs:** a synapse is signed by its presynaptic transmitter;
- **per-connection signs:** H.2's table, which corrects 342 of the atlas's signs, AWC → AIY among
  them.

## The cell and the substrate

**The cell** is block V's thermal cell at target 35, the cell of Logbooks 078 and 082:

- the connectome PPO brain on Cook 2019;
- settling to depth 4, block V's point otherwise;
- 3,000 episodes per run.

**The arms:** sign model {random, atlas, per-connection} × wiring {wild type, chemical-only null},
each learning and frozen. That is **12 arms.**

- **Grounded signs are held under PPO** (`enforce_synapse_signs`).
- **A null's edges carry the sign of the wild-type edge whose place they took.** Under the atlas
  model this is the same as signing by presynaptic transmitter.

**Each grounded level is calibrated to random signs' operating point** (design Decision F). Grounded
signs are mostly excitatory and saturate the settled network. At the draw's scale the calibration
measures 35.0% of units saturated under the atlas and 24.4% under per-connection signs, against
random signs' 8.5%. A first, uncommitted check on the pilot's seeds read 30–34%, 18–22% and 6%.

- **The fix:** `chemical_weight_scale` multiplies every chemical weight. One scale per level, shared
  by both wirings, matches the fraction of saturated units at initialisation to random signs'.
- **The calibration** ([scale-calibration.json](scale-calibration.json)) ran on seeds 9601–9616 under
  a rule fixed before it ran. It chose:
  - the atlas at **0.35**: 8.7% saturated, against a target of 8.5%;
  - per-connection signs at **0.42**: 8.7% saturated.
- **Not matched:** the action-mean spread is 0.41 at both grounded levels, against random's 0.29.
  It is reported beside.

| stem suffix (`connectomeppo_small_continuous2d_thermal_klinotaxis` + …) | signs | wiring | scale |
|---|---|---|---|
| `_t35`, `_frozen_t35` | random | wild type | 1.0 |
| `_rewired_chemical_null_t35`, `…_frozen_t35` | random | chemical-only null | 1.0 |
| `_atlas_signs_t35`, `…_frozen_t35` | atlas | wild type | 0.35 |
| `_rewired_chemical_null_atlas_signs_t35`, `…_frozen_t35` | atlas | chemical-only null | 0.35 |
| `_per_connection_signs_t35`, `…_frozen_t35` | per-connection | wild type | 0.42 |
| `_rewired_chemical_null_per_connection_signs_t35`, `…_frozen_t35` | per-connection | chemical-only null | 0.42 |

**Seeds 2201–2250 (50), 600 runs.**

## Gates

**Per sign level, in the gate preflight's shape:**

- each wiring's learner beats its frozen floor;
- the two learners do not both reach the 90% saturation bar.

A level failing either leaves the rung **unreadable.**

## The readings

Three readings on the wild type's lead `gap = wild type − chemical-only null`, on `auc_success`,
paired by seed:

| reading | definition | asks |
|---|---|---|
| **lead** | `gap(per_connection)` | does the chemical wiring matter under real signs? |
| **grounding** | `gap(per_connection) − gap(random)` | does grounding the signs move the lead? |
| **receptor** | `gap(per_connection) − gap(atlas)` | do the 342 corrected signs move it? |

Each is classified by `mc.classify` on two-sided Wilcoxon q values, **Holm-adjusted across the
three**.

**The minimum is 0.058**, 2/3 of the thermal split's committed 0.0865: **a change the size of the
cell's largest committed wiring effect.** That is the registered framing, chosen at the spec review.
2/3 of the chemical lead itself would be 0.011, below what this panel can resolve.

| state | lead | grounding, receptor |
|---|---|---|
| `move_wt` | **wild type leads**: the chemical wiring matters by at least the minimum | **lead grows** by at least the minimum |
| `move_null` | **null leads** by at least the minimum | **lead shrinks** by at least the minimum |
| `below` | **lead below the minimum**, significant | **change below the minimum**, significant |
| `no_move` | **no lead at the minimum** | **no change at the minimum** |
| `unresolved` | unresolved at this sensitivity | unresolved at this sensitivity |

**What can be read from them:**

- **The grounding contrast** reads three things at once: the signs, PPO's sign hold (random signs
  have nothing to hold), and the weight scale (0.42 against 1.0). The scale equalises the operating
  point, but the contrast still includes it.
- **The receptor contrast** has none of these confounds: both of its levels are held and calibrated
  to the same target, so it reads sign placement alone.

**Reported beside, never read as verdicts:**

- the lead at the random and atlas levels, and `gap(atlas) − gap(random)`;
- `episodes_to_30pct_success`;
- each level's learners against their floors;
- grounded runs that never learn;
- each reading's achieved spread and MDE.

**Also beside:** B.3's two acute knockouts, re-read on a modulated connectome with per-connection
signs.

- **Arms:** `modulated_perconn` and its floor, on B.3's seeds 2001–2032, on the lawn cell.
- **Scale:** these arms run at scale 1.0, as B.3's atlas arms did, so they are comparable with B.3.
  They are therefore not at random signs' operating point.
- **What is compared:** each knockout against its own intact arm. Comparing knockout effects across
  the two substrates is description only.
- **Non-learning seeds:** uncalibrated grounded nets sometimes never learned on the thermal cell. B.3's
  atlas lawn arms all learned, but any beside seed that does not learn is reported, and its
  knockouts are not read.

## The pilots

**First pilot, uncalibrated** (seeds 9501–9504; [pilot.json](pilot.json)):

- **Failures:** 4 of 16 grounded learning runs never learned (plateau under 5%; a fifth reached
  5.3%), with or without the sign hold (a
  6-run check with enforcement off).
- **The lead,** on the seeds that learned: per-connection +0.10 to +0.14, atlas +0.05, random −0.03.

**Second pilot, calibrated** (seeds 9511–9514; [pilot2.json](pilot2.json)), configured as the panel:

- **Failures:** 1 of 16 grounded learning runs never learned (atlas wild type, seed 9514).
- **The lead:**
  - per-connection **−0.070** [−0.121, −0.019];
  - atlas −0.075;
  - random +0.030.
- **The first pilot's per-connection lead does not survive calibration.** This is consistent with it
  having been an operating-point effect, which the calibration exists to remove.

## Sizing

The MDE is `2.487 × sd / √n`, at the second pilot's spreads:

| reading | pilot mean | pilot sd | seeds to reach 0.058 |
|---|---|---|---|
| lead | −0.070 | 0.088 | 15 |
| grounding | −0.100 | 0.146 | 40 |
| receptor | +0.006 | 0.164 | **50** |

**The panel is sized for the receptor reading, the narrowest question: 50 seeds.** A grounded run
that never learns swings its seed's lead by about ±0.3. The second pilot had one in 16, and that
risk is carried into the spreads.

## Before launch

- **Gate preflight** (`gate_preflight.py --panel sign_rung` on the second pilot,
  [preflight.txt](preflight.txt)): every level is `readable`, with plateaus 54–76% against the 90% bar.
  `launch: true`.
- **Cost:**
  - the second pilot's 48 runs took 49 minutes on 16 workers, so the panel's 600 runs take about
    **10 hours**;
  - the beside arms, 64 lawn runs, take about 2 hours more.
- **Seeds:** the panel's (2201–2250) are disjoint from both pilots (9501–9504, 9511–9514) and from
  the calibration (9601–9616).
- **Spec review:** `/nematode-review-spec add-per-connection-signs-rung`, before launch.

## Launch

From a worktree at this commit:

```bash
git -C ../nematode-d1 checkout --detach <this commit>
cd ../nematode-d1
P=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis
cfgs=()
for s in t35 frozen_t35 rewired_chemical_null_t35 rewired_chemical_null_frozen_t35 \
  atlas_signs_t35 atlas_signs_frozen_t35 rewired_chemical_null_atlas_signs_t35 \
  rewired_chemical_null_atlas_signs_frozen_t35 per_connection_signs_t35 \
  per_connection_signs_frozen_t35 rewired_chemical_null_per_connection_signs_t35 \
  rewired_chemical_null_per_connection_signs_frozen_t35; do cfgs+=(--config "${P}_$s.yml"); done
uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds 2201-2250 --runs 3000 --workers 16 \
  --output-dir ../nematode/campaigns/m10-panel \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# The beside arms, after the panel:
L=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_modulated_perconn
uv run python scripts/run_campaign.py --config ${L}.yml --config ${L}_frozen.yml --seeds 2001-2032 \
  --runs 3000 --workers 16 --output-dir ../nematode/campaigns/m10-b3 \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Then, from the main checkout:
rsync -a --ignore-existing ../nematode-d1/experiments/ experiments/
rsync -a --ignore-existing ../nematode-d1/exports/ exports/
uv run python scripts/analysis/sign_rung.py score --logs campaigns/m10-panel/logs \
  --seeds $(seq 2201 2250) --out-dir /tmp/m10-score --out panel.json --csv per-seed.csv
uv run python scripts/analysis/lawn_states.py --logs campaigns/m10-b3/logs \
  --arms modulated_perconn modulated_perconn_frozen modulated_perconn+tph1 modulated_perconn+pdfr1 \
  --learner-arm modulated_perconn --floor-arm modulated_perconn_frozen --minimum 0.278 \
  --knockout-minimum 0.009 --knockout-measure speed --seeds $(seq 2001 2032) --episodes 30 \
  --out b3-perconn.json --csv b3-perconn-per-run.csv
```

## Retention (A.0)

**Committed:**

- this record, both pilots' readings, the calibration and the preflight;
- the panel's `panel.json` and `per-seed.csv`;
- the beside arms' readings.

**Archived off-repo:** the campaign directories and the worktree's session records.
