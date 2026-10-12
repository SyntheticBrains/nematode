# M.9 — the thermal body cell: registration and launch

**Registered 2026-10-13, before any panel run.** Change: `add-thermal-body-cell`. Stage 1, its retry,
the connectome pilot, the point-worm anchor and the gate preflight are committed beside this record.

## The question

**Does the wild type's gap-junction placement advantage survive embodiment?** On the point worm, on
the thermal cell at target 35, the wild type leads a null whose gap junctions alone are rewired by
+0.096 `auc_success` (Logbook 082). M.9 asks the same question through the kinematic body.

The gap-only null is primary: a null must differ from the wild type in the layer under question (D21
as scoped 2026-10-12). The chemical-only null is read beside.

## The cell and the substrate

**The cell** is the thermal cell through the kinematic body, at **target 20** and **6,000 episodes**:

- the connectome PPO brain on Cook 2019, settling to depth 4;
- the thermotaxis block of Logbooks 078 and 082 (Tc 20, gradient 1.25, zone deltas and damage);
- 500 steps per episode;
- `body_model: kinematic`, `allow_reversal: true`, `action_space: body_drive`;
- `entropy_coef` 0.004 for the connectome arms;
- temperature, its derivative and the zone are read at the body's head, and the left–right
  difference comes from the synthetic head-sweep.

**Why target 20 and 6,000 episodes** (design Decisions C and G):

- **Stage 1** gated an MLP positive control at targets 35 and 20 over 3,000 episodes (seeds
  9701–9708). It read `fallback` at both, so by its fixed rule M.9 stopped. At 20, six of eight seeds
  learned (58–87%), but two started late and were still climbing
  ([stage1/](stage1/)).
- **The retry** was registered as a new decision, not a re-reading: target 20, 6,000 episodes at
  every stage, new seeds 9721–9728, the same rule. It **passed**: every seed competent, plateaus
  57–95% against 0% floors, +81.6 points over the floor [77.2, 86.1] ([retry/](retry/)).

**The arms:** wiring {wild type, gap-only null, chemical-only null} × {PPO, frozen}. That is
**6 arms.**

| stem (`connectomeppo_small_continuous2d_thermal_klinotaxis_body_` + …) | wiring | learning |
|---|---|---|
| `wt_ppo_t20`, `wt_frozen_t20` | `wild_type` | PPO, frozen |
| `gaponly_ppo_t20`, `gaponly_frozen_t20` | `rewired_gap_junctions_only` | PPO, frozen |
| `chemnull_ppo_t20`, `chemnull_frozen_t20` | `rewired_chemical_only` | PPO, frozen |

**Seeds 2301–2332 (32), 192 runs.**

## Gates

**Per null, in the gate preflight's shape:**

- each wiring's learner beats its frozen floor;
- the two learners do not both reach the 90% saturation bar.

**What a failing level means:**

- if the **gap-only** level fails either gate, the rung is **unreadable**;
- if the **chemical-only** level fails, only its beside reading is unreadable.

A level that saturates in the panel reads unreadable. It is not re-run at another target.

## The reading

**One registered reading:** the wild type's lead over the gap-only null,
`gap = wild type − gap-only null`, on `auc_success` over all 6,000 episodes, paired by seed.

- It is classified by `mc.classify` on the two-sided Wilcoxon q, with no family correction.
- **The minimum is 0.058** (0.0577), 2/3 of the thermal split's committed 0.0865, as M.8 read it.

| state | verdict |
|---|---|
| `move_wt` | **the gap advantage survives embodiment**: the wild type leads by at least the minimum |
| `move_null` | **the null leads** by at least the minimum |
| `below` | **a lead below the minimum**, significant |
| `no_move` | **no lead at the minimum**: embodiment washes the gap advantage out |
| `unresolved` | unresolved at this sensitivity |

**Reported beside, never read as verdicts:**

- the wild type against the chemical-only null (D21 requires it);
- the frozen floors' own gap, the wild type against each null;
- `episodes_to_30pct_success`;
- each learner against its floor, and any learner that never learns;
- the MLP's plateau (the retry) against the connectome's;
- C.3's kinematics for the wild type and the gap-only null, trained and frozen;
- the point-worm anchor and the horizon check, below;
- the reading's achieved spread and MDE.

## The pilot and the anchor

**Connectome pilot** (seeds 9711–9714, configured as the panel; [pilot/](pilot/)):

- **Gates:** every learner beats its 0% floor on every seed. Plateaus:

  - the wild type 86.1%;
  - the gap-only null 82.0%;
  - the chemical-only null 73.2% (one seed at 33.9%).

- **The lead over the gap-only null is +0.147 `auc_success`**, on 4 of 4 seeds (+0.080 to +0.241,
  sd 0.067).

- **The two wirings crawl alike.** The C.3 instruments match within about 1%, trained and frozen:

  - frequency 0.30 Hz;
  - wavelength 0.65 body lengths;
  - speed 0.11 body lengths/s;
  - reversal fraction 0.002 trained, 0.21 frozen.

  So at the pilot the lead is not a locomotion difference.

**The point-worm anchor at target 20 and 6,000 episodes saturates.** Both wirings plateau at 97–99%,
and the wild type's lead is −0.046 (+0.042, +0.001, −0.012, −0.214). It cannot pair with the body
reading. The point-worm comparison remains +0.096 at target 35 over 3,000 episodes, a different cell.

**The horizon check (Decision G).** The body's learners take a median 1,397 episodes to reach 30%,
the point worm's 307: a ratio of **4.6**, not the 2 the doubled budget assumed. If a lead in episodes
scales with learning time, a body lead in `auc_success` over 6,000 episodes is about 2.3 times the
same relative lead over the point worm's 3,000. So the pilot's +0.147 corresponds to about +0.064 on the point worm's scale. This is
description only; the verdict is read at 0.058 as registered.

## Sizing

The MDE is `2.487 × sd / √n`. At the pilot's spread (0.067), 0.058 is reached at about 9 seeds, so the
rule's floor is 16.

**The panel runs 32 seeds** (chosen with the user):

- the pilot's spread comes from four seeds;
- C.1e's spread through the body was 0.175, 2.6 times larger;
- at that spread, 32 seeds still resolve 0.077.

## Before launch

**Gate preflight** (`gate_preflight.py --panel thermal_body` on the pilot,
[pilot/preflight.txt](pilot/preflight.txt)) reads **`near_bar`** on both levels, so `launch: false`.

The wild type's plateau, 86.1%, is within the 5-point margin of the 90% bar. **This is accepted, with
the user, on these grounds:**

- **The gate fails only if both learners reach the bar.** The gap-only null sits at 82.0%, and its
  highest seed at 85.3%.
- **The reading is `auc_success`**, which is 0.45–0.79 at the pilot, far from its ceiling.
- **A ceiling would bias against the hypothesis.** It would compress the wild type's lead toward
  `no_move`, not manufacture one.
- **The alternative was costly and data-informed.** Moving to a harder target, as Logbook 078 did,
  would re-run the MLP control and the pilot (about 5 hours), with the MLP at risk of failing, and
  would choose the target after seeing this one's lead.

A level that saturates in the panel reads unreadable, as above.

**Cost:** the pilot's 40 runs took 2.4 hours on 16 workers. The panel's 192 runs take about **12
hours**.

**Seeds:** the panel's (2301–2332) are disjoint from:

- stage 1 (9701–9708);
- the retry (9721–9728);
- the pilot (9711–9714);
- every earlier panel.

**Spec review:** `/nematode-review-spec add-thermal-body-cell`, before launch.

## Launch

From a worktree at this commit:

```bash
git -C ../nematode-d1 checkout --detach <this commit>
cd ../nematode-d1
P=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis_body
cfgs=()
for w in wt gaponly chemnull; do for l in ppo frozen; do cfgs+=(--config "${P}_${w}_${l}_t20.yml"); done; done
uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds 2301-2332 --runs 6000 --workers 16 \
  --output-dir ../nematode/campaigns/m9-panel \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Every run must exit 0 with all 6,000 episodes before scoring: the scoring horizon is the shortest
# run's length, so one cut-short run would shorten it for every seed. The logbook reports the horizon.
# Then, from the main checkout:
rsync -a --ignore-existing ../nematode-d1/experiments/ experiments/
rsync -a --ignore-existing ../nematode-d1/exports/ exports/
uv run python scripts/analysis/thermal_body.py score --logs campaigns/m9-panel/logs \
  --seeds $(seq 2301 2332) --target 20 --out-dir /tmp/m9-score --out panel.json --csv per-seed.csv
uv run python scripts/analysis/thermal_body.py kinematics --logs campaigns/m9-panel/logs \
  --seeds $(seq 2301 2332) --target 20 --out kinematics.json
```

## Retention (A.0)

**Committed:**

- this record;
- stage 1, the retry, the pilot and the anchor;
- the panel's `panel.json`, `per-seed.csv` and `kinematics.json`.

**Archived off-repo:** the campaign directories and the worktree's session records.
