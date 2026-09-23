# A.2 — the calibration-and-robustness surface: registration and launch

**Registered 2026-09-21, before any panel seed ran.** Change: `add-operating-point-surface`.

Phase 8 **A.2**, roadmap decision **D16** as amended 2026-09-20. This record fixes the protocol
before the PPO campaign launches: the design, the metric rule, the sensitivity, the minimum effect
in both directions, and every branch the outcome can take. The pilot on seeds 109–112 has run and
is reported here; **no seed in either panel band has been touched.**

## The question

Does block V's wiring effect hold across the region of the learner's settings, or is it a fact about
one point in it?

The question is not rhetorical here. [Logbook 068](../068-l1b-rate-calibration.md) found one
inherited pin setting the **sign** of a registered primary: the width-by-wiring interaction ran
**+0.2818** at `plasticity_rate` 0.001 and **−0.0657** at 0.0001, a three-way of +0.3475 at q = 0.000
on 81 of 96 seeds, and the effect the pin hid was larger than the effect it was pinned for.
Churchland et al. 2026 (arXiv:2609.07355) report the same operating-point sensitivity in a connectome
reservoir independently. Every Phase 8 rung after this one cites a point that A.2 establishes.

## The design

One factor at a time around the committed operating point, on **hard350**, wiring crossed with each
pin level in turn.

| | PPO half | reading half |
|---|---|---|
| learner | `learning_rule` unset (PPO) | `readout_only` + `plasticity_eligibility: eprop` |
| `readout_width` | `per_neuron` | `per_neuron` |
| `forward_pass_depth` | 2, 3, 6 | 2, 3, 6 |
| `initial_log_std` | −1.0, −0.5, +0.5 | −1.5, −0.5, 0.0 |
| `plasticity_rate` | — (not read under PPO) | 1e-4, 1e-2 |
| `trace_decay` | — (not read under PPO) | 0.5, 0.99 |
| arms | 32 | 40 |
| seeds | **161–176** | **177–192** |
| runs | **512** | **640** |

**One cell, and the choice is measured rather than assumed.** A.1's committed per-seed data puts
thermal's interaction spread near 1,044 against a 134-episode wiring effect, so its detectable
interaction ran 1.6× to 3.4× the observed effect even at 32 seeds; hard350 came in at 0.50–0.68.
A sweep asks whether a sign moved, and on thermal that cannot be answered at any affordable size.
**Thermal runs only at levels where hard350 shows the sign move**, as a confirmation rather than as
a second surface.

**A frozen floor runs wherever a pin moves the untrained prior, and not elsewhere.** The three
construction pins change the arm before any learning, so each of their levels carries its own floor.
`plasticity_rate` and `trace_decay` cannot reach a frozen arm — it performs no updates — so the
centre's floor is theirs. That is the same runs read twice, not a missing control, and it is
asserted by test rather than argued.

**The centre is re-run fresh inside each campaign** rather than reused from block V or A.1, so every
interaction is a within-campaign contrast on one seed set. This runs a baseline alongside the
manipulation, which principle 6's 2026-09-19 note warns against, so the difference is stated: that
note forbids reading an ablation against a baseline the same campaign establishes at a *moved*
operating point. Here the centre **is** the committed point, unchanged, and its effect has been
measured across V.1, V.3, V.4 and A.1. If the centre fails to reproduce it on fresh seeds, that is
the finding and the surface is not read.

## The primary is the interaction; the trigger is the sign

Two different quantities, both registered, neither standing in for the other.

- **Primary at each level** — the **interaction**, `delta[s] = gap_at_level[s] − gap_at_centre[s]`,
  paired by seed, through the committed `paired_seed_wilcoxon_bootstrap`, BH-FDR across the pin
  family within a half. Required by the rule that a manipulation crossed with a structure contrast
  is read as an interaction and never as a main effect: a pin that moves both wirings equally
  cancels here, which is the point.
- **Trigger** — the **sign of the wiring gap** at that level, and whether its interval still
  excludes zero on the side it took at the centre. This is the quantity D16 names, and it is what
  decides whether a pin earns a full crossing.

A pin main effect is reported beside the interaction and never in place of it.

## The metric rule, fixed before the panel's rates are known — and scoped per level

`episodes_to_30pct_success` is right-censored at the horizon, and an interaction is a difference of
differences, which the metric rule forbids unless censoring is comparable across the cells that
contrast spans. So the **rule** is registered here and the **choice** follows the data: censoring
counted per arm, never pooled, and where the rates differ by more than **0.10** the uncensored
`auc_success` carries the interaction with the censored metric reported beside it. Both are reported
either way.

**The choice is made per level, not once for the surface, and the pilot is why.** The cells an
interaction spans are the centre and that level — not the campaign. Pooling the comparison across
every level lets one badly-censored level void the censored metric where it is perfectly
interpretable. The pilot found exactly that: at `forward_pass_depth: 2` the wild type never crosses
the threshold and the rewired null always does, a spread of 1.00, and under a pooled rule that one
level would have moved the primary at every other level too.

This refinement was made **after** seeing the pilot's rates and **before** any panel seed ran, which
is what a pilot on disjoint seeds is for. It is recorded as a change of scope rather than presented
as the original rule, because per-level is more favourable to the censored metric and a reader is
entitled to see that it was chosen on the requirement's wording rather than on convenience.

## Sensitivity, from a frozen source

The interaction requirement forbids computing a panel's sensitivity from the campaign's own results.
It comes instead from **A.1's committed hard350 per-seed CSV** — prior committed data, frozen
2026-09-21 — with the minimum detectable interaction at `2.487 × sd / sqrt(n)`.

| source spread | sd | MDE at n = 16 | ÷ centre effect |
|---|---|---|---|
| `episodes_to_30pct_success`, narrower | 659.5 | 410.0 | **0.71** |
| `episodes_to_30pct_success`, wider | 900.8 | 560.1 | **0.96** |
| `auc_success`, narrower | 0.0700 | 0.0435 | **0.71** |
| `auc_success`, wider | 0.0715 | 0.0445 | **0.73** |

The centre wiring gap on hard350 is **+580.44 episodes** and **+0.061** on `auc_success`.

**What that buys and what it does not, stated now rather than after the fact.** A **sign move**
requires an interaction at least as large as the centre effect, and the panel detects that at every
level. A **partial erosion** below roughly 0.7 of the centre effect is **not** resolvable at 16
seeds. So a level whose interaction falls below its MDE is reported as **unresolved at this panel's
sensitivity**, never as "no effect" — the distinction A.1 was made to draw and this panel inherits.

Sixteen seeds is matched to the registered decision rule, which is the sign, and is not matched to
measuring the surface's shape precisely. That is a deliberate scope choice: D16 asks which pins earn
a full crossing, and the crossing pass is where precision is bought.

## The registered minimum, in both directions

As a fraction of the **centre wiring gap measured in this campaign**, not of block V's published
figure, since the interaction is formed against the in-campaign centre.

- **A sign move**: the interaction's interval excludes an effect of the centre's own size, in the
  direction that would cancel it. That level earns a full crossing.
- **An erosion short of a sign move**: an interaction at or beyond **2/3** of the centre effect,
  significant after BH-FDR. Registered in both directions — a level that **amplifies** the wiring
  effect by 2/3 or more is the same finding with the opposite sign and is reported as such, not
  discarded as a nuisance.
- **Below the minimum**: reported as below it, licensing nothing on its own.
- **Unresolved**: the interval spans the minimum in either direction. Reported as unresolved with
  the panel's sensitivity beside it.

## Read separately, never pooled

The two halves are read separately and are **not** combined into one surface. `initial_log_std` in
particular is a different quantity on each: under PPO it is the start of a **trained** parameter,
under the rule it never trains and is the arm's fixed exploration noise for its whole run. A split
between the halves is reported as a split.

## The branches this campaign's outcome feeds

1. **No level moves the sign.** The committed operating point is not special, every later rung cites
   this surface, and no full crossing runs. Reported with the panel's sensitivity, because an
   underpowered null is not a clean null.

2. **A level moves the sign.** That pin earns a **full crossing**, registered separately (below),
   and thermal runs at that level as a confirmation.

3. **A level moves the sign on the PPO half.** Additionally, **A.1 is re-read at that point** — a
   registered outcome, not a surprise, per the tracker's operating-point requirement. Block V's
   verdict stands as read at its own setting and the condition is added as a dated note at every
   site that cites block V or Logbook 070, in the same sentence as the claim.

   **Amended 2026-09-21, after the pilot and before the panel read out**, because the clause is
   ambiguous in the case the pilot says is likely, and resolving it afterwards would be a choice
   made with the answer in hand.

   The tracker says A.1 is re-read "at the new point". That wording presumes A.2 **relocates** the
   operating point. But this change's Decision F says A.2 declares no winner and may not ablate
   against the point it establishes, so where the sign moves only at an extreme of a pin, there is
   no new point — nobody proposes running a later rung at `forward_pass_depth: 2`. The two rules
   are consistent only under a distinction, and it is drawn here rather than later:

   - **The re-read is owed** when the surface shows the committed point is **unrepresentative of
     the region a later rung would plausibly run at** — the sign moves at or adjacent to the
     committed setting, or across enough of the region that the committed point is a special case.
   - **The re-read is not owed** when the sign moves only at an extreme that no rung would adopt,
     while the committed point sits on a plateau. That outcome **bounds** block V rather than
     conditioning it, and the bound is recorded as a standing condition at the citation sites
     without re-running A.1.

   Either way the surface is reported in full and the bound is stated in the same sentence as the
   claim. What this amendment fixes is only whether 768 runs of A.1 are re-spent, and the answer
   now depends on where the sign moves rather than on how the result reads when it arrives.

4. **A level fails its learning gate.** Reported as a gate failure, which is itself a sensitivity
   result about that setting, and not worked around.

**`forward_pass_depth: 2` is named in advance.** On four pilot seeds the rewired null beat the wild
type on every one, by −0.17 to −0.28 on `auc_success`, and the wild type never reached the 30%
threshold while the null always did. Four seeds decide nothing and none of them is a panel seed.
But branch 2 and branch 3 are the likely readings at that level, and saying so now is the difference
between a registered outcome and a result that arrives looking like a discovery.

## The crossing pass is a second registration

D16 decides the full crossing **from** the one-factor pass, while the interaction requirement demands
a sensitivity frozen **before** the campaign and explicitly not computed from its own results. These
reconcile only one way: the crossing runs as its **own campaign with its own pre-registration**,
whose sensitivity is drawn from this pass once this pass is committed — at which point it is prior
committed data, which is the source the requirement names. Recorded here so a reviewer does not have
to reconstruct it.

## A.2 establishes a point and does not ablate against it

The operating-point requirement's last scenario forbids reading an ablation against the calibrated
baseline inside the campaign that establishes it, and D16 says the result is a sensitivity surface,
**not a new pin**. A.2 therefore declares no winner. Which point a later rung runs at is that rung's
calibration decision, citing this surface.

## The reading half owes drift evidence, or it is void

The reading learner leaves `w_chem` fixed, so the requirement governing a contrast under a learner
that does not write the wiring applies in full: the fixed tensors are compared against a control in
which nothing learned, **on every scored seed**, and any non-zero drift — or evidence missing for any
seed — returns **void**, not "it held".

This is an obligation on the campaign and not only on the write-up. `--track-experiment` is therefore
**not optional** on the reading campaign: without it no export path is recorded and the reader
returns nothing for every run. The weights auto-save is unconditional, so `--no-detailed-export`
stays safe. The centre's frozen arm is the correct comparator at the rate and decay levels, since
`w_chem` at a given seed is the same draw whatever those pins say and the learning arm never writes
it.

## Launch

```bash
# PPO half — 512 runs, seeds 161-176
uv run python scripts/run_campaign.py \
  $(for s in $(uv run python -c "import sys;sys.path.insert(0,'scripts/analysis');import operating_point_surface as o;print(' '.join(o.stem_for('ppo',a,l) for l in ['centre',*[x[1] for x in o._levels('ppo')]] for a in o.arms_at('ppo',l)))"); \
    do printf -- "--config configs/scenarios/foraging/%s.yml " "$s"; done) \
  --seeds 161-176 --runs 3000 --workers 16 \
  --output-dir campaigns/a2-ppo \
  -- --theme headless --track-experiment --no-detailed-export

# Score
uv run python scripts/analysis/operating_point_surface.py \
  --campaign campaigns/a2-ppo --half ppo --out-dir build/a2-ppo \
  --out build/a2-ppo/surface.json --csv build/a2-ppo/per-seed.csv
```

The reading half runs afterwards with `--half reading --seeds 177-192`, on the same flags.

## Artefact retention (A.0)

Committed: the parsed per-seed CSV, the analysis JSON, this launch record. Archived off-repo: the raw
campaign logs. The campaign directory is removed only after the CSV is committed, and any field that
cannot be compared because its source is gone is named **uncompared** rather than counted as
matching.

## Cost

The pilot ran **64/64 in 3,260 s at 15.5×** — learning arms at a median of 14.9 min, frozen at 10.9.
The PPO half is 512 runs at that mix: **roughly 6.9 hours**. The reading half's arms are dearer and
invert, the frozen one costing more than the learning one because an arm that never converges runs
every episode to the full budget: 20.0 min learning against 33.4 frozen, so 640 runs come to
**roughly 16.9 hours**. Detailed export is off; disk stays under 400 MB per campaign.
