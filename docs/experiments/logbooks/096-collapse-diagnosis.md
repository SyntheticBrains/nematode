# 096: Late Collapse Leaves No Registered Signature Before It Falls (Phase 9 M.1(a))

**Status**: completed. **No cause is indicated.** All six registered causes read `not_indicated`: in
the 1,500 episodes before each collapse, the collapsed body learners change the way their intact twins
on the same seed do.

Logbook 092's thermal body panel lost eight learners late in training. Each was re-run with read-only
per-update telemetry beside the other wiring's intact run on the same seed, and every re-run reproduced
its committed plateau. Against the intact twins' envelope, no cause's pattern departed in more than one
of the eight collapses.

| cause | pattern (each a conjunction of departures) | collapsed | intact | reading |
|---|---|---|---|---|
| weight growth → saturation → rank loss | weight norm up, saturated fraction up, `srank` down | 0 / 8 | 0 / 8 | `not_indicated` |
| a vanishing signal | raw advantage std down, gradient norm steady | 0 / 8 | 0 / 8 | `not_indicated` |
| stale second moments (sustained) | gradient norm up, then KL up | 0 / 8 | 0 / 8 | `not_indicated` |
| critic failure | explained variance down, value loss up | 0 / 8 | 0 / 8 | `not_indicated` |
| variance collapse | `log_std` down | 1 / 8 | 0 / 8 | `not_indicated` |
| mean-head saturation | the `\|u\| > 2` fraction up, `log_std` steady | 0 / 8 | 0 / 8 | `not_indicated` |

**The MLP control collapses on none of its 8 seeds** at 6,000 episodes (95% Wilson [0.00, 0.32]),
against 092's committed 2 of 32 for the wild type and 6 of 32 for the gap-only null.

**Beside it, never read as causes:**

- **Collapsed and intact runs drift alike.** Over the window, both sides' entropy rises by about 10,
  `log_std` by about 0.4 and the weight norm by about 4, and neither side's critic explains much.
- **Every update is clipped, on every learner.** The pre-clip gradient norm runs at 8–17 against the
  0.5 clipping threshold, and every minibatch of every update is clipped, for the connectome and the
  MLP alike, so each step is in effect a normalised gradient of fixed size.
- **Exploration grows through training.** The connectome's `log_std` climbs from about 0 to 1.1, the
  MLP's from 1.1 to 2.0, on runs that collapse and runs that do not.
- **Around onset**, the collapsed runs' critic-to-actor gradient ratio is lower than their twins' on 8
  of 8 seeds (median 1.0 against 1.8), and they show more transient gradient-then-KL spikes in the
  window on 7 of 8 (median 164 against 112). The first is read inside the fall itself, so it may be a
  consequence; neither is a registered cause.

**Date**: 2026-10-17.

**OpenSpec change**: `add-collapse-instrument`.

**Pre-registration**: [supporting/096-collapse-diagnosis/launch.md](supporting/096-collapse-diagnosis/launch.md),
committed at 02da3e4b before any re-run, with the cause set, the envelope, the departure rule and the
indicated rule fixed before any telemetry was read. D25's collapse definition was amended by the
maintainer on 2026-10-17, also before any re-run (below).

## The question

Late collapse of small PPO learners is documented at this network scale and horizon (Moalla et al.
2024; Elsayed et al. 2024; Dohare et al. 2024), and its candidate mechanisms leave different traces
before the success rate falls. Logbook 092's body panel lost two wild-type learners and six gap-only
learners after episode 3,800. **What happens inside a body learner in the 1,500 episodes before it
collapses, and does it happen in its intact twin?** A diagnosis, licensed by an identity check; it
chooses no remedy.

## Method

**The re-runs.** Logbook 092's configs, Cook 2019, the body at target 20, 6,000 episodes, with
`--update-telemetry --checkpoint-every 250` added: the eight collapsed runs (wild type 2313, 2328;
gap-only null 2304, 2305, 2307, 2308, 2319, 2330), each paired with the other wiring's intact run on the
same seed, which shares the seed, the task draw and every initial chemical weight; and the MLP control's
eight retry seeds (9721–9728). 24 runs, every one exiting 0. On these runs' own configs, a probe before
launch found telemetry and checkpoints change nothing: the episodes, the results CSV and every final
weight are identical with the flags on and off.

**Identity.** All 24 re-runs reproduce their committed final-quarter plateau to the precision it was
printed at ([identity.json](supporting/096-collapse-diagnosis/identity.json)). The step-level logs are
archived off-repo, so the plateau is the field compared; every other field is uncompared.

**Collapse** (D25, `scripts/analysis/collapse.py`, as amended): a fall begins where the trailing-100
success rate drops below half the highest trailing-200 rate seen so far and stays below for 300
episodes; a fall the rate climbs back from (above 0.8 of its peak) is a dip; the onset is the first fall
not recovered before the horizon. All eight collapsed re-runs collapse under it, at episodes 3,858–5,667,
and no intact twin does, so no pair was excluded.

**The diagnosis.** Every statistic is taken per episode, smoothed over 100 episodes, and read as its
change from the run's own baseline (the 1,500 episodes before the window). The window is the 1,500
episodes ending 100 before onset. The envelope is the intact twins' mean ± 2 sample sd at each aligned
episode, with a floor; a run departs when it sits outside in the predicted direction for 200 consecutive
episodes. Each intact twin is tested against the other seven. A cause is **indicated** when it departs in
at least 6 of 8 collapsed runs and at most 1 of 8 intact twins, **not indicated** when it departs in at
most 2 collapsed runs, and **mixed** otherwise.

## Results

### The registered reading

No cause departs in more than one collapsed run, and none departs in an intact twin
([diagnosis.json](supporting/096-collapse-diagnosis/diagnosis.json)). Every cause reads `not_indicated`.

Per statistic, the collapsed runs' departures are rare and split by direction: KL and clip fraction up
on 3 of 8 (down on 1), the saturated fraction up on 2, the gradient norm up on 2 and down on 1, and the
weight norm, entropy, explained variance, the critic-to-actor ratio and the `|u| > 2` fraction on none.
In the last 500 window episodes the collapsed and intact sides move by about the same amount on every
statistic: the weight norm by +4.2 and +4.0, `log_std` by +0.40 and +0.39, entropy by +10.0 and +9.8.

### The MLP's incidence

The MLP control collapses on 0 of 8 seeds (one recovered dip), against 092's committed 2 of 32 and 6 of
32 for the two connectome wirings ([mlp.json](supporting/096-collapse-diagnosis/mlp.json)). The 16
connectome re-runs, chosen half on collapse, read 8 of 16 under the definition, which only confirms that
it finds the eight known collapses.

### D25's definition, amended before the re-runs

On the archived 085 and 092 panels, the definition as first written (the onset at a run's first fall)
counted 35 of 64 wild-type runs on 085 against 092's own rule's 6, most of them dips early in learning
that the run recovered from, and placed gap-only seed 2305's onset at an early dip. The maintainer
amended it before any re-run: the onset is the first fall not recovered. Amended, the archived incidence
sits close to 092's rule (085's wild type 7 of 64; 092's wild type 3 of 32, its gap-only null 6 of 32),
and all eight known collapsers' onsets fall on their late collapse. The launch record carries both
readings.

## What the reading licenses

**M.1(b)'s remedy order.** With nothing indicated, M.1(b) tries D25's remedy list in its own order, as
registered: KL early stopping; weight clipping relative to initialisation; Adam ε 1e-4 and β₂ 0.99;
EWMA advantage normalisation; a learning-rate schedule last. Each is chosen by rescue of these eight
collapsers, read by discordant pairs, and must then hold the MLP control.

**What it suggests, never read as a cause.** Collapse here is not preceded by any slow drift the
registered causes describe; the two sides drift alike until the fall. That is what a sudden event would
look like, a bad update or a run of them, rather than a gradual degradation. Two platform-wide facts sit
beside it: every update is clipped, and the policy's exploration noise grows through training on every
learner. Neither separates collapsed from intact runs, and neither is on D25's list; changing either is
a registered decision, not this logbook's.

**The collapse statistic ships.** Every panel over 3,000 episodes now reports collapse incidence under
the amended definition, with Wilson intervals, Kaplan–Meier time to collapse, and the primary metric to
3,000 and to the horizon (`collapse.py`).

**Limits.** Eight collapsed runs for six causes; one cell, the body at target 20, on Cook 2019 and the
count-weighted gap model that Logbook 095 found carries an artefact. The intact envelope is eight runs,
most of them the other wiring (six wild type, two gap-only). The context around onset is read inside the
fall. The telemetry these runs wrote counted the `|u| > 2` fraction over action elements rather than over
steps, as the spec defines it; mean-head saturation was read on that element fraction, which is lower
on a many-output body drive. The telemetry now counts steps.

## Data

- [launch.md](supporting/096-collapse-diagnosis/launch.md): the registration, the readiness review and the D25 amendment.
- [probe.json](supporting/096-collapse-diagnosis/probe.json): telemetry on and off, identical, on these configs.
- [identity.json](supporting/096-collapse-diagnosis/identity.json): the 24 plateaus against the committed ones.
- [diagnosis.json](supporting/096-collapse-diagnosis/diagnosis.json), [stacks.csv](supporting/096-collapse-diagnosis/stacks.csv): the cause readout, the context around onset, the transient-spike counts, and every statistic's aligned smoothed series every 25 episodes.
- [mlp.json](supporting/096-collapse-diagnosis/mlp.json): the MLP's incidence.
- [incidence-085.json](supporting/096-collapse-diagnosis/incidence-085.json), [incidence-092.json](supporting/096-collapse-diagnosis/incidence-092.json): the archived panels under the amended definition.

```bash
uv run python scripts/analysis/collapse_diagnosis.py identity --logs campaigns/m1-diagnosis/logs \
  --logs campaigns/m1-mlp/logs --out identity.json
uv run python scripts/analysis/collapse_diagnosis.py diagnose --logs campaigns/m1-diagnosis/logs \
  --out diagnosis.json --csv stacks.csv
uv run python scripts/analysis/collapse_diagnosis.py mlp --logs campaigns/m1-mlp/logs \
  --logs campaigns/m1-diagnosis/logs --out mlp.json
```

The campaigns' raw logs, telemetry, checkpoints and session records are archived off-repo.
