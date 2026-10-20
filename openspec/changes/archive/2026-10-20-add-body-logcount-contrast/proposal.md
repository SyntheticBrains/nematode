## Why

On the corrected gap model (log-compressed counts), the point worm shows no wiring effect
([Logbook 099](../../../docs/experiments/logbooks/099-logcount-contrast.md)). The wild type leads its
gap-only null by +0.003 [−0.009, +0.016], against a minimum of 0.0577.

The phase's STOP condition reads: *the corrected model shows no wiring effect on either cell under PPO*.
Its two cells are the point worm and 092's body cell. The point worm has now read, and the body cell
decides. 092 was sequenced after M.1 because D25 forbade a 6,000-episode body panel before M.1 read.
M.1 has now read: its stop rule fired, and late collapse follows the training stream, not the seed
([Logbook 098](../../../docs/experiments/logbooks/098-collapse-remedies.md)).

**What 092's logs say about the horizon.** Read from the archived logs over its 32 seeds:

| window | mean lead | per-seed sd |
|---|---|---|
| episodes 1–3,000 | +0.115 | 0.242 |
| episodes 3,001–6,000 | +0.137 | 0.291 |

Two consequences follow:

- Under the unmodified learner every observed collapse began after episode 3,858. A verdict read to
  3,000 episodes is almost entirely free of its stream-driven noise.
- The late half carries as much of 092's lead as the early half. Dropping it would leave a null result
  open to the objection that the effect is late.

On 2026-10-19 the maintainer chose the **hybrid**. The panel runs 6,000 episodes, the verdict is read on
the first 3,000, and the late window is reported beside it with the collapse statistic. The maintainer
also chose **128 seeds**, frozen floors on 16 seeds, and no count-weight arm.

## What Changes

- **Configs**: 092's four body configs (wild type and gap-only null, learning and frozen; target 20,
  `body_model: kinematic`), with two keys changed: `connectome_source: emmons_2024_hermaphrodite`, per
  D23, and `gap_weight_mode: log_count`. They come from a new generator,
  `generate_body_logcount_configs.py`, and a loader test re-reads them.
- **The panel**:
  - wild type and gap-only null learning on seeds **4001–4128** (128);
  - both frozen on the first 16;
  - 6,000 episodes, **288 runs**.
- **The gate pilot**: all four arms on seeds 9895–9898, configured exactly as the panel (16 runs). Smokes
  on 9899–9900, at 20 episodes.
- **One registered reading**: `gap(log, body)` on `auc_success` over **episodes 1–3,000**, wild type −
  null, paired by seed. It uses an 80% bootstrap interval and the two-sided Wilcoxon as its q, a family
  of one, and is classified by `mc.classify` at **m = 0.0577**. The verdict map is fixed before any run
  and names the joint outcome with the point worm:
  - `no_wiring_effect` here takes the **STOP branch**;
  - `placement` puts the question of moving G.1–G.3b to the body to a dated decision.
- **Beside, never deciding**:
  - the same gap over all 6,000 episodes and over 3,001–6,000;
  - D25's collapse statistic per arm (incidence with Wilson intervals, Kaplan–Meier time to collapse);
  - plateaus at both horizons;
  - episodes to 30% success;
  - the D24 audit.
- **The harness** `scripts/analysis/body_logcount_contrast.py` reads success series directly, so that
  the primary and the gates are computed on the first 3,000 episodes. It has four subcommands, `plan`,
  `preflight`, `table` and `score`, and reuses Logbook 099's verified manifest.
- **Spec** (`architecture-comparison-protocol`): a panel longer than the collapse hazard may fix its
  verdict on a window that ends before the hazard begins, with the rest described.

No brain or environment code changes.

## Impact

- **Cost**, from 092's own run records on this cell (125 minutes a learning run, 36 a frozen one, at 16
  workers): the panel about **34.5 hours**, the pilot about 2 hours. The pilot measures it again on these
  configs, and the launch record carries the pilot's figure.
- **Seeds**: new bands 4001–4128 (panel), 9895–9898 (pilot) and 9899–9900 (smokes), added to the
  seed-band table.
- **Downstream**:
  - `no_wiring_effect` → the STOP branch: the phase's deliverable becomes the statement that the
    project's one surviving wiring effect belonged to the electrical-synapse model.
  - `placement` → a dated decision on the mechanism block.
  - Any other verdict → a dated decision.
  - Then the decomposition of Logbooks 074–092 is restated.
