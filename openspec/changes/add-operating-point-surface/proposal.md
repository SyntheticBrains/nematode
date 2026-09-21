## Why

Phase 8's **A.2**, the milestone that fixes the operating point every later rung cites.

[Logbook 068](../../../docs/experiments/logbooks/068-l1b-rate-calibration.md) found **one inherited
pin setting the sign of a registered primary**: the width-by-wiring interaction ran **+0.2818** at
`plasticity_rate` 0.001 and **−0.0657** at 0.0001, with a three-way of +0.3475 at q = 0.000 on 81 of
96 seeds. The effect the pin hid was larger than the effect it was pinned for. Churchland et al.
2026 (arXiv:2609.07355) report the same operating-point sensitivity in a connectome reservoir
independently, which is why the roadmap calls this deliverable publication-critical rather than
housekeeping: any reservoir-style claim from this project will be read against that paper.

**D16**'s rule follows: no Phase 8 contrast runs at a pin unswept for the learner it uses. It was
**amended 2026-09-20** because B.1 and B.2 both register PPO arms "at A.2's swept point", which the
original wording blocked. So A.2 has two halves.

**Reading the pins first changed the design, and two of the findings are corrections.** The five
pins are not five of a kind:

- **`readout_width` is categorical** — `pooled` or `per_neuron`, nothing between. Its sweep is one
  alternative level, not a grid.
- **`plasticity_rate` and `trace_decay` are inert under PPO.** `plasticity_rate` reaches the rule
  only on the non-PPO branch; `trace_decay` matters only when `enable_activity_traces: true`. A PPO
  config may legally set either, and 100 committed configs set `plasticity_rate`. It does nothing.
  **An arm that looks swept and is not is the failure this change adds a requirement against.**
- **`initial_log_std` measures different things on the two halves.** Under PPO it is the start of a
  *trained* parameter; under the reading learner it never trains and is the arm's fixed exploration
  noise for its whole run. One surface cannot speak for both.
- **Two of the three PPO pins have never been varied on this substrate at all.**
  `forward_pass_depth` is `4` in 149 of 149 connectome configs, and `initial_log_std` has only a
  two-point hand-sweep on MLP node-perturbation arms. Logbook 069's "never varied" is exact.

**A correction to the tracker's own wording.** It calls the three pins "shared by both learners".
That holds for the PPO arm *of `ConnectomePPOBrain`*, which is what block V is. It does not hold for
the MLP yardstick: `readout_width` and `forward_pass_depth` are not declared on `MLPPPOBrainConfig`,
and a YAML setting them under `mlpppo` is **dropped with a warning, not an error**. A later rung
citing A.2's surface onto an MLP arm would be citing a sweep that never happened.

## What Changes

### 1. A sensitivity surface, one factor at a time, in two halves

Wiring {wild type, rewired null} crossed with each pin level in turn, around the committed point.

| | PPO half | reading half |
|---|---|---|
| learner | `learning_rule` unset (PPO) | `readout_only` + `plasticity_eligibility: eprop` |
| `readout_width` | `per_neuron` | `per_neuron` |
| `forward_pass_depth` | 2, 3, 6 | 2, 3, 6 |
| `initial_log_std` | −1.0, −0.5, +0.5 | −1.5, −0.5, 0.0 |
| `plasticity_rate` | — (inert) | 1e-4, 1e-2 |
| `trace_decay` | — (inert) | 0.5, 0.99 |
| arms | **32** | **40** |
| runs, 16 seeds | **512**, ≈ 7.4 h | **640**, ≈ 16.9 h |

**A frozen floor is run wherever a pin changes the untrained prior, and not elsewhere.**
`readout_width`, `forward_pass_depth` and `initial_log_std` change what the arm is before any
learning, so each of their levels needs its own floor. `plasticity_rate` and `trace_decay` cannot
move a frozen arm — it performs no updates — so the centre's floor serves their levels, which is
the same runs read twice rather than a shortcut. That claim is asserted by test, not argued, and it
saves 4.4 hours: on the reading learner the **frozen** arm is the dearer one, 33.4 min against 20.0,
because an arm that never learns never converges and every episode runs the full budget.

The **PPO half runs first**: it is what B.1 and B.2 cite, and the only half that can force an A.1
re-read.

### 2. One cell, and the reason is measured rather than assumed

**hard350 carries the sweep; thermal runs only if hard350 shows a sign move.** A.1 measured
thermal's interaction spread at ~1,044 against a 134-episode effect, so its detectable effect ran
1.6× to 3.4× the observed one even at 32 seeds. A sign-change question is unanswerable there at any
affordable size. hard350 came in at 0.50–0.68. Spending half the compute on readings that cannot
resolve would buy nothing.

### 3. The primary is an interaction; the trigger is a sign

Each level's primary is the **interaction** of pin level with wiring, paired by seed against the
campaign's own centre — per the requirement that a manipulation crossed with a structure contrast is
read as an interaction and never as a main effect. What D16 calls for separately is the **sign of
the wiring gap** at that level, and that is the registered trigger for a full crossing. Both are
reported; neither stands in for the other.

### 4. The crossing pass is a second registration

D16 decides the full crossing *from* the one-factor pass, while the interaction requirement demands
a sensitivity frozen before the campaign. These reconcile only if the crossing runs as its own
campaign whose sensitivity is drawn from the one-factor pass as prior committed data. Stated in the
launch record, because a reviewer will look for it.

### 5. A generator, because fifty-odd configs is not a hand-authoring job

There is no config-override mechanism in this repo — no `extends`, no `--set`, and the campaign
runner passes one passthrough to every run, so it could not vary a pin across arms anyway. Every
grid point needs a committed YAML. A generator in the mould of `scripts/campaigns/l4_panel_pilot.py`
emits them from their committed parents, one key each, and a test asserts the one-key delta.

### 6. The instrument is not touched, and each half uses the door built for it

Scoring goes through the **unmodified** `wiring_premise.py` and `connectome_structure_efficiency.py`.
A new driver builds manifests and reports the surface, in the mould `init_sharing_control.py`
established.

The two halves enter by different doors, because `wiring_premise`'s arm map is keyed on `wt_ppo` and
`rn_ppo` and the family around it is block V's PPO panel. The **PPO half** takes that path, which is
A.1's. The **reading half** calls the efficiency instrument directly with its own wild and rewired
labels — the path `l4_rate_calibration.py` took for L.1b, the only previous sweep on this learner.
Routing `three_factor` arms through a PPO-keyed map would mean mislabelling them to suit a function
signature.

### 7. The reading half owes drift evidence, or it is void

The reading learner leaves the chemical matrix fixed, so the requirement governing a contrast under
a learner that does not write the wiring applies in full: the fixed tensors are compared against a
control in which nothing learned, **on every scored seed**, and missing evidence returns **void**
rather than "it held". That makes `--track-experiment` non-optional on the reading campaign, since
without it no export path is recorded and the drift reader returns nothing for every run. The
weights auto-save is unconditional, so suppressing detailed export stays safe.

## Capabilities

**Modified**: `architecture-comparison-protocol` — **one** added requirement: a swept level is shown
to reach the learner it is set on.

One, not several. The requirements that already govern this campaign — interaction reading, power in
advance when a null carries a consequence, the learning gate per arm, **the fixed-substrate contrast
rule and its drift clause**, metric departure, committed-baseline reuse, the operating-point re-read,
standing conditions, and A.1's sharing-by-test rule — are cited in `design.md` rather than
restated. A.4 has just finished
redistributing 38 rules that accumulated one per rung, and the inherited discipline is to add a
requirement only where no existing one reaches.

The case none reaches: **a pin can be accepted, validated and ignored.** `plasticity_rate` on a PPO
arm is a no-op; `readout_width` under `mlpppo` is dropped with a warning. Either produces an arm
that looks swept, scores cleanly, and establishes nothing — and a sweep is the one experiment whose
entire output is a claim about pins.

## Impact

**Code:**

- `scripts/campaigns/generate_operating_point_configs.py` — new generator
- `scripts/analysis/operating_point_surface.py` — new driver

**Tests:**

- `packages/quantum-nematode/tests/quantumnematode_tests/analysis/test_operating_point_surface.py` — new
- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_connectome_pin_reach.py` — new

**Configs:** roughly 57 generated, committed YAMLs under `configs/scenarios/foraging/`, each exactly
one key off its parent, using the established suffix vocabulary. Committed arms serve their levels
**unchanged** wherever they already exist — the four block-V arms as the PPO centre, and eleven
`eprop` arms across the reading half's centre, `_wide` and `_r1e4` levels. The generator skips a
config that exists and the delta test checks the committed ones too, so a committed arm that turns
out to differ in more than its named key is caught rather than assumed.

**Docs:** `openspec/changes/phase8-tracking/tasks.md` (A.2, and the dated MLP-scope correction);
`docs/roadmap.md`'s A.2 rows; a logbook at the close.

## Breaking Changes

None. No production code path changes; the work is configs, a generator, a driver and tests.

## Backward Compatibility

Every committed config and every prior result is unaffected. The centre configs state the pins'
defaults explicitly, and a test asserts that stating a default explicitly is byte-identical to
leaving the key absent — so the centre arm is the committed operating point and not a near-miss of it.
