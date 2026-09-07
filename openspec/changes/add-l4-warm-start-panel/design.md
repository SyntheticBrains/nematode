# Design: the warm-start panel

## Context

Every earlier panel started its arms from random weights and measured a distribution of fixed
points seeded by those weights. This panel starts every arm from the same competent policy,
cloned per seed, so that the seed-to-seed variance the panels could not overcome is what the
clone carries, not what the initialisation draws. The teacher is the MLP-PPO champion of
Logbook 029 (89% at 6000 episodes); no trained weights of it exist, so it is trained first, and
the campaign runner's auto-save (`exports/<session>/weights/final.pt`) supplies the file.

## Goals / Non-Goals

**Goals**

- Measure whether the connectome holds a competent policy on both wirings (representability).
- Measure what the reward-modulated rule does from that policy, against the same floors panel 1
  used, on the same paired seeds.
- Measure whether a warm start makes PPO on the connectome competitive (D13).

**Non-Goals**

- Tuning the clone or the rule to the warm start; every hyperparameter is fixed in advance.
- The MLP yardstick, count-scaled initialisation, or any arm the earlier panels retired.

## Decisions

### D1. Teacher and recording

Train `mlpppo_small_continuous2d_combined_klinotaxis.yml` on seeds 1–8 at 6000 episodes through
the campaign runner. The teacher is the seed with the highest committed plateau tail; its
`final.pt` is copied to `campaigns/l4-warm-start/teacher.pt`. Recording: a derived config beside
the results with `freeze_updates: true` and `weights_path` pointing at the teacher, run for 300
episodes at seed 101 (disjoint from every panel seed) with `--record-rollouts`; the teacher's
frozen plateau tail at that seed is recorded as the ceiling every clone is read against. Its
action means, not its samples, are the cloning targets.

### D2. Clones

For each seed 1–8 and each wiring, two clones from the one rollout file, by
`scripts/campaigns/l4_behavioural_clone.py` with hyperparameters fixed here: 300 epochs, learning
rate 1e-3, batch 256, holdout 0.2 by episode.

| clone | student config | parameter set | file |
|---|---|---|---|
| plastic-set, wild-type | the plastic frozen arm | `plastic` (chemical weights behind the anatomical readout) | `clones/plastic_wt_seed{seed}.pt` |
| plastic-set, rewired | the plastic frozen rewired-null arm | `plastic` | `clones/plastic_rn_seed{seed}.pt` |
| full-set, wild-type | the low-noise PPO arm (`_lowstd`, the PPO config plus `initial_log_std: -1.0`) | `full` (chemical weights, gains, readout; the noise is saved at −1.0, untrained) | `clones/full_wt_seed{seed}.pt` |
| full-set, rewired | the low-noise PPO rewired-null arm | `full` | `clones/full_rn_seed{seed}.pt` |

Every clone's initial, final and held-out loss is recorded in `clones.json`; a clone whose
held-out loss does not fall below half its initial is flagged, not excluded. The student's seed
is the run seed, so a clone loads into exactly the brain whose initial weights it started from;
for the rewired arms that is also what keeps them paired — a clone made at seed S draws the same
rewiring as the arm at seed S and is refused by the wiring check at any other seed, which the
tests exercise.

**Why the full-set student is low-noise.** The trainer leaves the noise parameter untrained but
saves it, so a clone carries its student's `initial_log_std` into whatever arm loads it. Built
from the PPO config that would be 0 (std 1.0), and the full-set frozen arm would be evaluated at
the noise panel 1 showed caps every arm — incomparable with the plastic-set frozen clone at −1.0.
The full-set student is therefore built from a one-key derivation of the PPO config with
`initial_log_std: -1.0`, and the from-scratch PPO comparator for W6 is that same derived config
without weights, so both PPO arms start at the same noise and the clone is their only difference.
Logbook 029's original PPO arm (noise 0, 6000 episodes) is cited descriptively, not run.

### D3. Per-seed weight paths

`weights_path` accepts `{seed}`, which the entry point substitutes with the run seed before
loading; a path with braces left after substitution is an error. A relative path is resolved
against the working directory, and the campaign runner runs from the repository root, so the
clone paths below are repository-relative. This is the only package-side
change and it is inert for every existing config.

### D4. Arms, seeds, budgets

Twelve arms on paired seeds 1–8 (`rewire_seed` from the run seed; every clone made at that
seed), three budgets, the committed plateau detector deciding the single registered extension
(a fresh run at 1.5× replacing the shorter log):

| key | config | start | rule | budget |
|---|---|---|---|---|
| `wt_clone_frozen`, `rn_clone_frozen` | plastic frozen arms + `weights_path` | plastic-set clone | frozen | 600 |
| `wt_clone_hebbian`, `rn_clone_hebbian` | plastic Hebbian arms + `weights_path` | plastic-set clone | unmodulated Hebbian | 2000 |
| `wt_clone_plastic`, `rn_clone_plastic` | plastic arms + `weights_path` | plastic-set clone | three-factor | 2000 |
| `wt_fullclone_frozen`, `rn_fullclone_frozen` | plastic frozen arms + `weights_path` | full-set clone | frozen | 600 |
| `wt_fullclone_ppo`, `rn_fullclone_ppo` | low-noise PPO arms + `weights_path` | full-set clone | PPO | 3000 |
| `wt_ppo`, `rn_ppo` | the low-noise PPO arms (`_lowstd`) | random | PPO | 3000 |

**Configs** (twelve new, under `configs/scenarios/foraging_predator_thermal/`, stem
`connectomeppo_small_continuous2d_combined_klinotaxis`): `<stem>_plastic_frozen_clone`,
`<stem>_plastic_frozen_rewired_null_clone`, `<stem>_plastic_hebbian_clone`,
`<stem>_plastic_hebbian_rewired_null_clone`, `<stem>_plastic_clone`,
`<stem>_plastic_rewired_null_clone` (plastic-set clones, one `weights_path` key off their
parents); `<stem>_plastic_frozen_fullclone`, `<stem>_plastic_frozen_rewired_null_fullclone`
(full-set frozen, one key off the plastic frozen arms); `<stem>_lowstd`,
`<stem>_rewired_null_lowstd` (the PPO arms plus `initial_log_std: -1.0`, one key off their
parents; the from-scratch comparators and the full-set students); `<stem>_lowstd_fullclone`,
`<stem>_rewired_null_lowstd_fullclone` (one `weights_path` key off the low-noise arms).

The full-set clone's readout replaces the anatomical one where it is loaded; the arm key says
so. Every value the plastic arms run with is panel 1's pin; the PPO arms run the committed 029
recipe. Cost: 32 frozen runs at 600, 32 plastic/Hebbian at 2000, 32 PPO at 3000, plus eight
teacher runs at 6000 — roughly four hours on 16 workers. No pilot.

### D5. Metric and statistics

The committed plateau-tail full-clear success, the committed paired-seed one-sided Wilcoxon, 80%
bootstrap CI and BH-FDR, through panel 2's reader. Panel 2's committed per-seed table supplies
the random-initialisation frozen floor on seeds 1–8 for the gate.

### D6. The confirmatory family (six tests, one BH-FDR family at α = 0.05)

| id | test | direction | reads |
|---|---|---|---|
| **W1** | `wt_clone_frozen` vs panel 2's `wt_frozen`, seeds 1–8 | clone > random | the gate: the plastic-set clone holds a better-than-random policy |
| **W2** | `wt_clone_frozen` vs `rn_clone_frozen` | wild-type > rewired | the wild-type holds the teacher's policy better than its scramble (D13's structural question) |
| **W3** | `wt_clone_plastic` vs `rn_clone_plastic` | wild-type > rewired | **the primary**: the warm-started 2×2 |
| **W4** | `wt_clone_plastic` vs `wt_clone_frozen` | plastic > frozen | the rule improves on a competent start |
| **W5** | `wt_clone_plastic` vs `wt_clone_hebbian` | plastic > Hebbian | reward matters from a competent start |
| **W6** | `wt_fullclone_ppo` vs `wt_ppo` | warm-started > scratch | the warm start helps PPO on the connectome (D13) |

**Power at n = 8, stated in advance.** The exact one-sided Wilcoxon on eight paired seeds has a
smallest attainable p of 1/256; under BH over six tests the best-ranked test passes only at
p ≤ 0.0083, so a test passes only with seven or eight concordant seeds. The family is built to
detect large, consistent effects — which is what a competent start on every seed is meant to
produce — and the descriptive layer carries everything smaller. Eight seeds is the ratified
sample for cost (about four hours); a null here reads "not confirmed at n = 8".

A test passes at q < 0.05 with a positive mean delta. Reverses (interval entirely below zero)
are named. **Descriptive**: the full-set frozen pair; `wt_fullclone_ppo` vs `rn_fullclone_ppo`;
every clone arm against the teacher's ceiling; the from-scratch PPO pair; clone fit losses;
every remaining pair; per-seed sign counts.

### D7. The verdict map

In order: `insufficient_seeds` (any of W1, W3, W4, W5 with fewer than two common seeds);
`clone_fail` (W1 fails — the clone is not competent, and nothing downstream is interpretable);
`sanity_floor_fail` (W4 or W5 fails — the rule does not improve a competent start, or
reward-modulation adds nothing over alignment from it); `rewired_beats_wild_type` (W3's interval
entirely below zero); `specific_wiring` (W3 passes); `degree_statistics` (W3's interval spans
zero); `inconclusive`. W2 and W6 annotate and never change the verdict (`wild_type_holds_better`,
`warm_start_helps_ppo`). A further annotation, `rule_destroys_clone`, is set when W4's interval
lies entirely below zero: the answer the panels feared, that the rule takes a competent policy
apart.

Claim type: performance throughout.

### D8. Scripts and records

`scripts/campaigns/l4_warm_start.py`: `teacher` (select the best seed from the MLP campaign by
plateau tail, copy its weights, write the frozen recording config, run the recording), `clone`
(the 32 clones, `clones.json`); both write beside the campaign results.
`scripts/analysis/l4_warm_start.py`: the twelve-arm registry, seed range 1–8 enforced, panel 2's table for W1, the
family, the verdict, the annotations, the descriptive pairs and ceilings, clone fits, per-seed
CSV and curves. Tested on synthetic values: registry and seed range; every test's direction; the
family size; every verdict row in order; each annotation; the `{seed}` resolution; the teacher
selection on a synthetic campaign; the recording config derivation; the clone file naming.

## Risks / Trade-offs

- **The plastic-set clone may not be competent.** Chemical weights behind a fixed anatomical
  readout and random gains may not represent the MLP's policy. `clone_fail` is a registered
  outcome, and the full-set clone's frozen arm shows whether the limit is the parameter set or
  the wiring.
- **The rule may take the clone apart.** Panel 1 saw a frozen 37% policy go to zero under the
  rule on one seed. `rule_destroys_clone` is registered so that result is named, not buried.
- **n = 8 again.** A competent start should collapse the bimodality that defeated the earlier
  tests; if it does not, the panel reports as before and the sample is the stated limit.
- **The teacher's visitation distribution is the dataset.** A clone can fit the teacher's
  visited states and still fail elsewhere; the frozen clone arms measure exactly that.

## Open Questions

None.
