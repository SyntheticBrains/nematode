# 014 supporting: Baldwin Inheritance Pilot — Details

Detail-level data and discussion for the M4 logbook. The main logbook
(`docs/experiments/logbooks/014-baldwin-inheritance-pilot.md`) contains
the headline findings and the GO/PIVOT/STOP decision; this file
contains per-seed tables, the evolved-hyperparameter distributions,
the F1 innate-only forensic discussion, and the wall-time breakdown.

## Per-seed final-fitness tables

**TBD — to be filled when the pilot completes.** Source: each arm's
`history.csv` final row, plus the F1 CSV.

## Per-seed gen-to-0.92 trajectories

**TBD — to be filled when the pilot completes.** Source:
`artifacts/logbooks/014/m4_baldwin_pilot/summary/convergence_speed.csv`.

## Evolved-hyperparameter distributions

**TBD — to be filled when the pilot completes.** Per-seed final
`best_params.json` decoded against the schema, with histograms of
each evolvable field across the 4 Baldwin seeds.

Notable for the M4 hypotheses:

- **`weight_init_scale`**: did TPE converge on 1.0 (the default —
  meaning the field is uninformative) or push toward the schema
  bounds [0.5, 2.0]? Risk 1 of the design doc explicitly canaries
  this.
- **`entropy_decay_episodes`**: did TPE converge on 500 (the brain
  default) or learn that a faster/slower decay produces better
  K-train convergence?

## F1 innate-only forensic discussion

**TBD — to be filled when the F1 evaluator completes.** Per-seed F1
success rates, comparison to the hand-tuned baseline mean, and
discussion of whether the genetic-assimilation gate's "+0.10pp over
baseline" threshold was the right calibration.

If F1 floor: discuss whether to recalibrate the gate ("+0.03pp" or
"F1 > random init") and whether the elite hyperparameters genuinely
encoded learning bias or just produced from-scratch policies that
happen to be no worse than baseline.

## Wall-time breakdown

**TBD — to be filled when the pilot completes.** Per-arm + per-seed
timings, comparison to M3's per-arm wall-time, impact of
`early_stop_on_saturation: 5` on the saturating arms.

## Schema confounder check

**TBD — to be filled post-pilot.** The Baldwin pilot evolves 6 fields
vs the M3 control's 4 fields. Did the speed gate (Baldwin vs control)
pass because of the new evolvable knobs, or because the control's
4-field schema was missing critical degrees of freedom that any
extra field would have provided? Cross-check the per-arm evolved
hyperparameters across the 4 retained fields.

## Cross-arm code-revision check

The Lamarckian and control reruns are on the M4 code revision; M3's
published numbers were on the M3 revision. Confirm the rerun numbers
reproduce M3's published numbers within seed-to-seed noise (the
weight_init_scale brain field defaults to 1.0 = byte-equivalent; the
kind() Protocol method is pure-additive; the early-stop monitor only
fires when the flag is set).

**TBD — comparison table here when the rerun completes.**
