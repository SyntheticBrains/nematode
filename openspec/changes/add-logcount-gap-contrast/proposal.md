## Why

G.0 read **`artefact`** ([Logbook 095](../../../docs/experiments/logbooks/095-gap-operating-point.md)):
the wild type's lead over the gap-only null on the thermal cell rode on the model's count-scaled gap
coupling. G.0-A then chose the corrected model by learnability alone
([Logbook 097](../../../docs/experiments/logbooks/097-gap-substrate.md)): **log-compressed counts**
(`gap_weight_mode: log_count`). Ohmic coupling was eligible but learned more slowly on both cells.

The corrected model's wiring contrast has not been read. G.0 measured its log-count level only as
16-seed description: +0.012 [−0.023, +0.046]. The pilot's harness was barred from reading the gap.
Block G's mechanism items (G.1–G.3b) and the phase's STOP branch both turn on that contrast. The STOP
branch applies if the corrected model shows no wiring effect on either cell under PPO.

The maintainer fixed the sequence on 2026-10-17:

1. The point-worm contrast on the corrected model runs first. Its seeds are 3901–3964 and its minimum
   is 0.0577.
2. 092's body contrast is re-run on the corrected model after M.1 reads out, because D25 forbids a
   6,000-episode body panel before then.

This change is step 1.

## What Changes

- **The contrast** is the wild type against the gap-only null on log-compressed counts, on the thermal
  cell at target 35 on Emmons 2024. It runs 64 seeds, 3901–3964, with each wiring learning and frozen:
  **256 runs** at 3,000 episodes. The configs are G.0-A's four `*_g0a_log_*_t35.yml`, unchanged, so the
  pilot that chose the model is also this panel's gate evidence on disjoint seeds.

- **One reading** is `gap(log)` on `auc_success`, wild type − null, paired by seed. It uses the one-sided
  Wilcoxon and an 80% bootstrap interval, and is classified by `mc.classify` at **m = 0.0577**. The
  verdict map is fixed before any run (design Decision B): `placement`, `below_minimum`,
  `no_wiring_effect`, `reversed`, `unresolved` or `unreadable`. A `placement` lead also serves as D23's
  anchor on the corrected model, which fixes G.1's minimum.

- **D24's matched statistics and shortcut audit** for both log-count wirings, on the panel's 64 draws.
  They are computed and committed before launch, with the existing `matched_statistics`.

- **The analysis harness** `scripts/analysis/logcount_contrast.py` has four subcommands:

  - `plan`: the configs and seeds;
  - `preflight`: the gates on the G.0-A pilot's log arms;
  - `table`: D24;
  - `score`: the reading.

  It refuses an incomplete panel, a run that did not exit 0, and a log that does not record the config
  its name claims.

- **Spec** (`architecture-comparison-protocol`): a wiring effect traced to a model artefact is re-read
  on the corrected model, under its own registration, before any mechanism work builds on it.

No brain or environment code changes.

## Impact

- **Cost**: about 4 hours at 16 workers. That is 128 learning runs at 22.9 minutes and 128 frozen runs
  at 5.1 minutes, at G.0's measured rates on the same cell and episodes. It runs after M.1(b)'s rounds
  release the machine.
- **Seeds**: 3901–3964, reserved by `add-ohmic-gap-coupling` in the seed-band table. No new smoke band
  is needed, because the configs already ran 44 smokes and 352 pilot runs.
- **Downstream**: the verdict decides whether G.1–G.3b register on log counts and at what minimum. It
  also decides whether the STOP branch stays live, pending the body contrast.
