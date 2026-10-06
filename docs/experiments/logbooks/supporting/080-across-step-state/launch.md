# B.2a — across-step state and its positive controls: registration and launch

**Registered 2026-10-06, before any scored run.** **Amended 2026-10-07: the first pilot left no τ eligible; the maintainer chose one recalibration of the input gain (below), then a repeat pilot on fresh seeds.** Change: `add-across-step-state`. The τ rule below was
committed with the change (`1cea31df`) before the pilot that applies it ran.

## The question

Every connectome result so far runs on a memoryless substrate: each environment step starts every
neuron at zero, injects the sensors and settles through four `tanh` updates. B.2a gives each neuron a
membrane potential that persists across an episode's steps, with gap junctions as ohmic coupling
between potentials and the sensors as a current held through the step. B.2b, plastic gap junctions
under PPO, is MUST and runs on this substrate only if it passes here.

**The question is whether PPO learns each block-V cell on the leaky substrate at least as well as on
the settling one.** What each answer licenses, per cell:

| verdict | meaning | licenses |
|---|---|---|
| **non_inferior** | the dynamical wild type learns within the cell's margin of the settling one, or better | B.2b runs on the leaky substrate there, after first re-reading the wild type's lead over the chemical-only null on it |
| **inferior** | it learns worse by more than the margin | B.2b on that cell closes *unreachable-with-reason*, the diagnosis its deliverable |
| **unlearnable** | the dynamical arm does not beat its own frozen floor | as `inferior` |
| **unreadable** | a gate the panel needs fails on the settling arm, or both arms saturate | neither; the record says which gate |
| **unresolved** | the interval spans the margin | neither; the maintainer chooses a registered extension on fresh seeds, or closing |
| **no_positive_control** | MLP-PPO does not reach competence on the cell | the cell's solvability is not shown; stop there |

The two cells are read separately; nothing is combined across them.

## The substrate

`dynamics: leaky`, one global `membrane_tau_steps`, `bptt_chunk_length: 16`. Within a step,
`forward_pass_depth` (4) semi-implicit Euler sub-steps of `τ dv/dt = −v − L v + Wᵀ tanh(v) + I`; leak
and the gap Laplacian implicit, so the step is stable on the raw gap weights, whose row sums reach 232
against at most 6.2 of chemical fan-in. PPO replays contiguous 16-step chunks from stored starting
potentials. **What differs from the settling substrate**, so that a verdict is read as one about the
substrate as built and not about memory alone: the potentials carry across steps; the sensors are a
held current, not the initial state; gap junctions act on potential differences, not as added drive;
and signal can cross the settling budget's hop limit by travelling across steps.

## Control 1 — the cell is solvable by the strongest method

MLP-PPO on each cell, **seeds 1201–1208**, 3,000 episodes. **Passes if every seed's plateau success is
at least 30%**, the episode metric's competence level.

- hard350: `mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64`, the existing
  width-64 config, whose environment is the connectome cell's exactly. (Logbook 060 committed only its
  per-width means, so the control is run here rather than cited.)
- thermal at target 35: `mlpppo_small_continuous2d_thermal_klinotaxis_t35`, the connectome cell with
  the existing thermal MLP-PPO brain.

## The time constant: pilot and rule

Candidates **0.2, 1 and 5 steps** (about 1, 5 and 25 worm-seconds at D22's default). On **seeds
1101–1104**, per cell: the dynamical wild type learning and frozen at each τ, and the settling wild type
learning and frozen, 64 runs. **The rule, fixed before the pilot ran:**

1. A τ is eligible on a cell if `gate_preflight` reads it `readable` there: both substrates beat their
   frozen floors and the higher plateau sits more than 5 points below the 90% bar.
2. Each eligible τ's score is the smaller of the two cells' mean paired difference, dynamical minus
   settling `auc_success`. **τ = 1 step is chosen** unless another eligible τ beats its score by more
   than **0.05**, about one pilot standard error at the proxy spread on four seeds; then that τ is
   chosen. If τ = 1 is not eligible, the eligible τ with the higher score is chosen.
3. If no τ is eligible on both cells, no panel launches.

No wiring gap is read: only the wild type runs.

**First pilot result (2026-10-07, unit input gain, seeds 1101–1104, 64/64 runs, [first-pilot.json](first-pilot.json)): no τ is eligible, so no panel launches
(rule step 3).** The dynamical wild type barely learns at any τ:

| τ (steps) | hard350 leaky / settling plateau | status | thermal leaky / settling plateau | status |
|---|---|---|---|---|
| 0.2 | 1.6% / 77.3% | readable | 0.0% / 69.9% | fails_floor |
| 1 | 0.9% / 77.3% | readable | 0.0% / 69.9% | fails_floor |
| 5 | 0.0% / 77.3% | fails_floor | 0.0% / 69.9% | fails_floor |

**Diagnosis, measured without training on the hard350 wild type at initialisation.** The leaky
substrate's steady state passes almost none of the sensory signal to the readout: the policy mean's
sensitivity to the food features is **1.4 × 10⁻³** at every τ (τ sets only how fast the steady state
is approached), against **0.6** for the settling substrate, about 400 times less. Sensor potentials
reach about 0.39, motor potentials about 0.004. Each hop has a gain below one: the leak pulls every
potential toward zero, the chemical weights are scaled for unit fan-in norm, and gap coupling shunts
toward neighbours; the settling map has no leak and does not attenuate. Raising the chemical gain does
not restore it: at four times the gain motor activity becomes large but self-sustained, and
input sensitivity stays near 10⁻². The design checked stability and replay but not this gain; protocol
principle 2's feasibility arithmetic would have caught it before the pilot.

**The recalibration (2026-10-07, [calibration.json](calibration.json)).** One new setting, `input_gain`,
a fixed multiplier on the sensor current under leaky dynamics, chosen without training on calibration
seeds 2001–2008 as the gain whose steady-state sensitivity ratio to settling is closest to 1 on the worse
cell. **512** (0.84 hard350, 0.90 thermal; 256 gives 0.44 and 0.56, 1,024 gives 1.27 and 1.22). The
recurrent gain stays at 1, below the critical gain of all 64 untrained wild types checked (lowest 1.09).
The τ pilot is repeated at input gain 512 on **seeds 1105–1108** under the same rule. **Stopping rule:**
if it also leaves no τ eligible, B.2a closes as a failed positive control and B.2b closes
*unreachable-with-reason*.

**Repeat pilot result (2026-10-07, input gain 512, seeds 1105–1108, 64/64 runs, [pilot.json](pilot.json)):
τ = 0.2 steps is chosen**, the only τ readable on both cells (rule step 2, τ = 1 being ineligible).

| τ (steps) | hard350 leaky / settling plateau | status | thermal leaky / settling plateau | status | mean `auc_success` difference (hard350, thermal) |
|---|---|---|---|---|---|
| 0.2 | 41.9% / 77.8% | readable | 26.9% / 68.2% | readable | −0.265, −0.205 |
| 1 | 28.2% / 77.8% | readable | 0.2% / 68.2% | fails_floor | −0.339, −0.283 |
| 5 | 0.1% / 77.8% | readable | 0.0% / 68.2% | fails_floor | −0.434, −0.283 |

Described, not a verdict: at τ = 0.2 the per-seed differences are −0.217, −0.069, −0.434 and −0.341
on hard350 and −0.295, +0.007, −0.277 and −0.255 on thermal; on four seeds both 80% intervals lie
below −δ. Learning falls monotonically as τ rises: the less the substrate carries across steps, the
better it learns.

## Control 2 — the panel

Per cell, the dynamical wild type, learning and frozen, at the chosen τ, paired by seed with the
committed settling wild type:

| cell | seeds | settling runs reused from | margin δ | source of δ |
|---|---|---|---|---|
| hard350 | 641–768 | `a3-boundary` (Logbook 079) | **0.0143** | 2/3 of A.6's committed lead over the chemical-only null |
| thermal, target 35 | 513–640 | `a6t2-thermal-split` (Logbook 078) | **0.0239** | Logbook 078's registered minimum on that cell |

**Reading**, on `auc_success`, `d = dynamical − settling`, paired by seed, 80% bootstrap interval:

| verdict | condition |
|---|---|
| **non_inferior** | lower bound above −δ |
| **inferior** | upper bound below −δ |
| **unresolved** | otherwise |

Gates first: the dynamical arm must beat its frozen floor (else `unlearnable`); the settling arm must
beat its floor and the two must not both reach 90% (else `unreadable`). Episodes to competence are
reported beside and never read.

**Sizing.** No committed run pairs the two substrates, so the spread is a proxy: the paired
wild-type-minus-null gap on the same seeds, sd 0.062 on hard350 (Logbook 079) and 0.10 on thermal
(Logbook 078). Two substrates may decorrelate a seed more than two wirings on one, so the proxy is
expected to err low. At 128 seeds and a true difference of zero, the chance of reading `non_inferior`
is about 0.91 on hard350 and 0.92 on thermal at the proxy spread. The achieved spread is reported
beside, never used to re-read.

## The reuse is licensed first

The first four learning and two frozen settling seeds of each band, 12 runs, are re-run on this
change's code with the committed command line, and compared with the committed logs: every `Run:` line
and the final `w_chem`, bit for bit (`across_step_control.py --identity`). **No panel run launches until
all 12 match.** The result is committed here as `identity.json`.

## Before launch

- **Gate preflight** on the chosen τ: *to be filled.*
- **Cost**, from the pilot's own run times at 16 workers: *to be filled.*

## Launch

*Commands to be filled with the chosen τ.*

## Retention (A.0)

Committed: this record, `first-pilot.json`, `calibration.json`, `pilot.json`, `identity.json`, `mlp.json`, the panel's `control.json` and
`per-seed.csv`. Archived off-repo: the pilot's, the identity check's, the MLP controls' and the
campaign's raw logs.
