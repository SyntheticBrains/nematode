# The warm-start panel: what a local rule does from a competent policy

## Why

Three panels (Logbooks 040–042) resolved the same way: on the C3 cell the random initial
weights decide the outcome, most random initialisations are dead, and the reward-modulated rule
neither beats its floors nor separates the wirings at the sample sizes run. The imitation warm
start (roadmap D13; tracker S.2) removes the confound the panels kept hitting: clone a competent
policy into the connectome on every seed, then ask what the rule does from there. It also answers
D13's original question — whether the connectome can *hold* a good policy at all, or whether its
fifth-of-six rank under PPO (Logbook 029) was a representability limit rather than a learnability
one — and whether a warm start makes PPO on the connectome competitive.

The tooling exists (PR #324, #325): the connectome brain saves and loads weights, a teacher's
rollouts can be recorded with its action means, and a student can be cloned into either of two
parameter sets. This change registers the experiment that uses it.

Ratified with Chris 2026-09-07: one teacher, the best of eight MLP-PPO seeds; two clones per
student — the plastic set the rule can inherit, and the full PPO set; the warm-started plastic
2×2 with its floors, plus PPO fine-tuning from the full clone against PPO from scratch.

## What Changes

- **Per-seed weight paths**: `weights_path` may contain `{seed}`, resolved with the run seed
  before loading, so one committed config addresses one clone file per seed.
- **Twelve configs**, each one key off an existing arm: the frozen, Hebbian and
  three-factor arms on both wirings warm-started from the plastic-set clone; the frozen arm on
  both wirings warm-started from the full-set clone; and the PPO arm on both wirings warm-started
  from the full-set clone; and a low-noise PPO arm on both wirings (`initial_log_std: -1.0`),
  which is both the full-set clone's student and the from-scratch comparator, so the warm-started
  and scratch PPO arms start at the same noise. Twelve arms.
- **The registration**: teacher selection and recording rules; clone hyperparameters fixed in
  advance and every clone's fit reported; twelve arms on paired seeds 1–8 at three budgets; a
  six-test BH-FDR family with the warm-started wild-type-over-rewired three-factor contrast as
  the primary, a clone-competence gate, two sanity floors and the D13 PPO contrast; an ordered
  verdict map; one bounded extension per run; no pilot.
- **A campaign script** (`scripts/campaigns/l4_warm_start_campaign.py`) that selects the teacher, records
  it frozen, and clones every student; **a harness** (`scripts/analysis/l4_warm_start.py`) that
  fixes the analysis before any data exist.
- The launch record, the runs, the records under `supporting/043-l4-warm-start/`, tests, docs.

Out of scope: any change to the rule, the substrate or the trainer; the logbook.

## Capabilities

**Modified**: `l4-plasticity-panel` (the warm-start panel), `weight-persistence` (the `{seed}`
placeholder).

## Impact

- Edited: `scripts/run_simulation.py` (placeholder resolution), ten new configs, two new scripts
  and their tests, the config variant tests, `docs/usage.md`, `configs/README.md`, `CHANGELOG.md`.
- The clone configs are not smoke-tested: they reference weight files a campaign produces.
- Default runs are byte-identical; nothing in the package changes.
