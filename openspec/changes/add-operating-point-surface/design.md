## Overview

A.2 registers the calibration-and-robustness surface. `docs/roadmap.md` § Phase 8 **D16**, as
amended 2026-09-20, is authoritative for what must be swept; `openspec/specs/phase8-tracking` carries
the Operating-Point Discipline requirement it enforces. This document records what the pins turned
out to be, why the design is one cell and two halves, and the scope decisions taken with the
maintainer before anything ran.

## Design Decisions

### Decision A: The five pins are not five of a kind, read off the code rather than the prose

| pin | declared | default | what a sweep of it can be |
|---|---|---|---|
| `readout_width` | `connectome_ppo.py:240` | `"pooled"` | **two levels only**, `pooled` or `per_neuron`. Not a grid |
| `forward_pass_depth` | `connectome_ppo.py:256` | `4` | integer; `>= 1` enforced at **construction**, not load |
| `initial_log_std` | `_plasticity_config.py:297` | `0.0` | float, unbounded; rejected only against `state_dependent` |
| `trace_decay` | `_plasticity_config.py:128` | `0.9` | float, `ge=0.0, lt=1.0`; needs `enable_activity_traces` |
| `plasticity_rate` | `_plasticity_config.py:135` | `0.01` | float, `gt=0.0` |

Three consequences the plan turns on:

1. **`plasticity_rate` and `trace_decay` are inert under PPO.** `plasticity_rate` reaches
   `ConnectomeThreeFactorRule` only on the non-PPO branch (`connectome_ppo.py:2298-2312`);
   `trace_decay` is read only when `enable_activity_traces: true`. Both are declared on the shared
   mixin, so a PPO config may set them and 100 committed configs set `plasticity_rate`. D16 already
   assigns them to the reading half; the code is why.
2. **`initial_log_std` is not "the PPO actor's".** It is a mixin field, and its meaning differs by
   learner: under PPO it is in `learnable_parameters` and trains from that start point; under the
   plastic rule it never trains, so it is the arm's fixed exploration noise for the whole run. The
   two halves are measuring different things about this pin and the write-up says so rather than
   presenting one surface.
3. **`readout_width` and `forward_pass_depth` exist only on `ConnectomePPOBrainConfig`.** Under any
   other brain they are dropped with a warning, not an error (`config_loader.py:200-217`).

### Decision B: The primary is an interaction; the sign is the trigger

D16 asks for "a full crossing only for pins that move the wiring effect's sign", which is a
statement about the **wiring gap at each level**. The governing requirement asks that a manipulation
crossed with a structure contrast be read as an **interaction**. These are different quantities and
both are registered:

- **Primary at each level** — the interaction, `delta[s] = gap_at_level[s] − gap_at_centre[s]`,
  paired by seed, through `paired_seed_wilcoxon_bootstrap`, BH-FDR across the pin family within a
  half. A pin that moves both arms equally is a fact about the pin, not about the wiring.
- **Trigger** — the sign of the wiring gap at that level, and whether its interval still excludes
  zero on the side it took at the centre.

Neither stands in for the other, and a pin main effect is reported beside the interaction and never
in place of it.

### Decision C: One cell, chosen on measured sensitivity

A.1's committed per-seed data settles this. On thermal the interaction spread ran ~1,044 against a
134-episode wiring effect, giving a detectable interaction of **1.6× to 3.4×** the observed effect
even at 32 seeds; on hard350 the same arithmetic gave **0.50–0.68**. A sweep asks whether a sign
moved, and on thermal that question cannot be answered at any affordable size — clearing the
registered bar there needs roughly 800 seeds.

So **hard350 carries the sweep**, and thermal runs only at levels where hard350 shows a sign move.
This is the second finding of Logbook 070 being spent rather than merely recorded.

### Decision D: A frozen floor wherever a pin moves the untrained prior, and not elsewhere

A wiring effect is claimed at each level, and the gating requirement makes each claim unreadable
unless its arm beats its own frozen floor on the same seeds. The floors also matter on their own
account: the frozen arm shows whether a pin moves the **untrained prior**, which separates "this pin
changes what the learner can do" from "this pin changes where the learner starts".

But the five pins divide. `readout_width`, `forward_pass_depth` and `initial_log_std` are
**construction** pins — they change the arm before any learning, so each level needs its own floor.
`plasticity_rate` and `trace_decay` are **learning-only**: a frozen arm performs no updates, so
neither can reach it, and the centre's floor is the correct floor for their levels. That is the same
runs read twice, not a missing control.

This saves 8 arms and 128 runs on the reading half, and the saving is larger than the run count
suggests: **a frozen reading arm is the dearer one.** Measured medians on hard350 are 20.0 min for
`eprop_readout_only` against **33.4 min** for `eprop_frozen`, because an arm that never learns never
converges, so every episode runs the full step budget. The 128 runs not taken are 4.4 hours. (The
PPO half runs the other way round — 16.7 min learning against 11.1 min frozen — where the update
itself is the larger share of the work.)

It is also exactly the kind of claim A.1 taught this programme to test rather than argue, so the
pin-reach test asserts it directly: at `freeze_updates: true`, changing `plasticity_rate` or
`trace_decay` leaves the run identical.

### Decision D2: Committed arms serve their levels unchanged

A.1's precedent is that the committed block-V configs served its baseline level unmodified, which
kept the panel-commit identity its test pins. The same applies here: the four block-V hard350 arms
are the PPO centre, and eleven committed `eprop` arms cover the reading half's centre, `_wide` and
`_r1e4` levels. Only the missing arms are generated.

**Nothing committed is a PPO `per_neuron` arm.** Every one of the 18 configs setting
`readout_width: per_neuron` is a `three_factor` arm, so the PPO half's `_wide` level is entirely
new — which is Logbook 069's "readout width swept only under the rule" showing up as a file count.

The delta test covers the committed arms as well as the generated ones. A committed arm that turns
out to differ from the centre in more than its named key is a config to regenerate, not a fact to
work around.

### Decision E: The centre is re-run fresh inside each campaign

The committed block-V arms and A.1's `edge_order` arms both sit at this operating point, but the
reuse rule permits reuse only under a parsed-field identity check with **no partial reuse**, and a
failed check forces a full re-run after spending the check. Running the centre inside each campaign
makes every interaction a within-campaign contrast on one seed set.

**This runs a baseline in the same campaign as the manipulation, which principle 6's 2026-09-19 note
warns against, so the difference has to be stated.** What that note forbids is reading an ablation
against a baseline the same campaign establishes at a *moved* operating point. Here the centre **is**
the committed point, unchanged, and its wiring effect has been measured across V.1, V.3, V.4 and
A.1. The in-campaign arm replicates a known baseline rather than establishing an unknown one. If the
centre fails to reproduce the effect on fresh seeds, that is the finding and the surface is not read.

### Decision F: A.2 establishes a point and does not ablate against it

The operating-point requirement's last scenario forbids reading an ablation against the calibrated
baseline inside the campaign that establishes it, and D16 says the result is "a sensitivity surface,
not a new pin". A.2 therefore declares no winner. Which point a later rung runs at is that rung's
calibration decision, citing this surface.

### Decision G: The crossing pass registers separately

D16 decides the full crossing from the one-factor pass; the interaction requirement demands a
sensitivity frozen before the campaign and explicitly **not** computed from the campaign's own
results. The only consistent reading is that the crossing is a **second campaign with its own
pre-registration**, whose sensitivity is drawn from the one-factor pass once that pass is committed —
at which point it is prior committed data, which the requirement names. Recorded in the launch
record rather than left for a reviewer to reconstruct.

### Decision H: The instrument is not modified, and the two halves reach it by different doors

`wiring_premise.py` hard-codes its cells, arms, family and minimum effects. Editing them to admit a
pin axis would forfeit the replication property V.4 and A.1 both rest on. A new driver,
`scripts/analysis/operating_point_surface.py`, sits in the mould `init_sharing_control.py`
established: an explicit stem-to-arm mapping built by a loop rather than a regex, a manifest
builder, a completeness gate, per-level scoring, and a surface reporter. Its test asserts both
instrument files are byte-identical to `main`.

**The two halves do not take the same route into it, and the reason is in the instrument's own
constants.** `wiring_premise.EFFICIENCY_ARMS` is `{"wt_ppo": …, "rn_ppo": …}`, and the `FAMILY`,
`MIN_EFFECT` and verdict apparatus around it is block V's PPO panel. The reading learner's arms are
`three_factor`, and labelling them `wt_ppo` in a manifest to get them through that door would be a
mislabel adopted for the convenience of a function signature.

- **PPO half** → `wiring_premise.efficiency_contrast`, which is A.1's path and the one whose
  replication property the PPO arms are entitled to.
- **Reading half** → `connectome_structure_efficiency.analyse` **directly**, with that module's own
  `_WILD` / `_REWIRED` labels. This is not a workaround: it is the path `l4_rate_calibration.py`
  took for L.1b, the only previous sweep on this learner, and it reaches the same four metrics, the
  same BH-FDR family and the same verdict rule.

Both instruments stay unmodified on either route.

### Decision H2: The reading half owes drift evidence on every scored seed, or it is void

The reading learner leaves `w_chem` fixed, so the requirement governing a wiring contrast under a
learner that does not write the wiring applies in full. Its first scenario is not a documentation
rule: the fixed tensors **SHALL** be compared against a control in which nothing learned, the
comparison **SHALL** cover every scored seed, and any non-zero drift — **or drift evidence missing
for any scored seed** — returns **void**.

This is an obligation on the campaign, not only on the write-up, and it is cheap to forfeit by
accident. The evidence is `<exports_path>/weights/final.pt`, read the way
`l4_reduced_perturbation._chemical_weights` reads it, compared against the frozen arm at the same
seed the way `l4_frozen_features.drift` compares it. Two consequences:

1. **`--track-experiment` is not optional on the reading campaign.** Without it no `exports_path` is
   recorded and the reader returns `None` for every run, which reads as "unavailable" and voids the
   half. The weights auto-save itself is unconditional (`run_simulation.py:1020`), so
   `--no-detailed-export` remains safe and the disk cost stays bounded.
2. **The shared frozen floor is the right comparator here too.** `w_chem` at a given seed is the
   same draw whatever `plasticity_rate` or `trace_decay` says, and the learning arm never writes it,
   so the centre's floor is the correct control for the rate and decay levels rather than a
   substitute for a missing one.

The PPO half owes none of this: PPO writes the chemical weights, so its contrast is not one run
under a learner that leaves them fixed.

### Decision I: One new requirement, deliberately

Nine existing requirements govern this campaign and are cited rather than restated: the interaction
reading; power stated in advance when a null carries a consequence; the per-arm learning gate; the
fixed-substrate contrast rule, whose drift clause Decision H2 discharges; the metric-departure rule;
committed-baseline reuse; the operating-point re-read; standing conditions; and A.1's
sharing-by-test rule.

The case none reaches: **a pin can be accepted, validated and ignored.** `plasticity_rate` on a PPO
arm passes its `gt=0.0` bound and is never read. `readout_width` under `mlpppo` is dropped with a
log warning. Either produces an arm that looks swept, scores cleanly, and establishes nothing. For
most experiments that is a curiosity; for a sweep, whose entire output is a claim about pins, it is
the failure mode. Added as a requirement that a swept level be shown to reach its learner.

Only one is added, per the discipline A.4's consolidation set.

## Open Questions (to resolve during implementation)

- **The levels themselves.** `forward_pass_depth` {2, 3, 6} brackets the canonical 4-hop pathway
  without reaching the degenerate K=1. `initial_log_std` differs by half because the two learners sit
  at different centres (0.0 under PPO, −1.0 under the rule) and each sweep must bracket its own. The
  pilot confirms the extreme levels run rather than fixing the grid.
- **Whether depth 2 learns at all on hard350.** If it does not, the gate fails and that level is
  reported as a gate failure rather than as a wiring reading — which is itself a sensitivity result.
- **Minimum effect.** Registered as a fraction of the **centre wiring gap measured in this
  campaign**, in both directions, with power from A.1's committed hard350 per-seed spread.

## Risks

- **The reading half is 18.5 hours.** It runs second, after the PPO half has read out, so a defect
  found in the shared generator or driver is caught on the cheaper campaign first.
- **80 committed configs is a large surface for a one-key-delta claim.** The generator plus the
  per-config delta test is what makes that claim checkable rather than asserted.
- **A pin level could move RNG consumption**, which is A.1's defect in a new place. The readout draw
  is documented as happening at the pooled shape whatever the width, so the stream is untouched —
  exactly the kind of claim A.1 proved must be tested. The brain test asserts the shared generator's
  post-construction state across pin levels rather than reading the comment.
- **`initial_log_std` non-zero against `continuous_std_mode: state_dependent` is rejected at load,
  and that validator is not repeated at construction**, so a `model_copy`-derived arm would skip it.
  The generator emits YAML, which takes the validated path, and the config test confirms each file
  loads.
