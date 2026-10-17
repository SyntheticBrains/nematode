## Context

G.0 found the gap-junction placement lead on the thermal cell is carried by how the model couples gap
junctions: additive drive scaled by EM counts (Logbook 095). Additive coupling has no biophysical
counterpart: an electrical synapse passes current in proportion to the potential difference across it.
The maintainer chose (2026-10-17) to build ohmic coupling into the settling map and select between it and
log-compressed counts in a gated pilot, then re-run the corrected model's contrasts: the point worm now,
the body after M.1 reads out.

## Decisions

### Decision A: Ohmic coupling as an implicit step inside each settling iteration

Today the settling map starts from `h₀ = s`, the injected sensors, and each of `K` iterations computes
`h ← tanh(W_chemᵀh + Gᵀh + c)`, `c` the modulator field's held current where present. Under
`gap_coupling: ohmic`:

```text
h₀ = M⁻¹ s,   then K times:   h ← tanh( M⁻¹ (W_chemᵀh + c) ),   M = I + L,   L = diag(G·1) − G
```

`M⁻¹ d` is the steady state of `dv/dt = −v + d − Lv`: within each iteration the gap-coupled potentials
equilibrate to the drive, each pair's current proportional to the difference of the two potentials and to
the pair's conductance. The injected sensors are equilibrated the same way before the first iteration, so
a sensor still relays across its gap junctions, as the additive map's first iteration relays `Gᵀs`.

**What the operator guarantees.** `L·1 = 0`, so `M⁻¹·1 = 1`; `M⁻¹` is symmetric and non-negative, so each
row is a weighted average of the drive (on Emmons 2024, eigenvalues 0.0032 to 1). **The gap coupling adds
no loop gain at any conductance**: a pair cannot hold activity through its gap junction, as the additive map
lets a pair coupled above about 1 do. The chemical recurrence is unchanged and can still sustain activity,
as it can today. `G` is fixed, so `M⁻¹` is computed once, in double precision, and registered as a
non-persistent buffer, as the leaky substrate's operator is; it is applied by one matrix product per
iteration (right-multiplied on a batch).

**The conductances are the brain's gap matrix as built**: the counts divided by `sqrt(d_i d_j)`, with
`gap_weight_mode` and `gap_weight_scale` applied. The form changes; at scale 1 the conductances do not,
and the pilot's ohmic arms run at the scale Decision D calibrates. That is not the
same as an unchanged operating point: 296 of 302 neurons share one gap component, so `M⁻¹` is dense, and on
Emmons the median neuron keeps 33% of its own drive (AVA 9%, AVB 13%). Decision D measures that before any
run. Rejected: the explicit diffusive form `tanh(W_chemᵀh − Lh + c)`, which is stiff at these
conductances (the largest eigenvalue of `L` is about 314) and oscillates; raw counts as conductances, which
would change the form and the magnitude together.

**Refusals**, by the validator and again at build (`model_copy` skips validators): gap junctions off;
`dynamics: leaky` (already ohmic); `plastic_gaps` (the operator is fixed); any learning rule other than
PPO; and, under PPO, node perturbation, activity traces and e-prop eligibility, which reach into the
settling loop and are defined on the additive map. The body drive, the critic, `forward` and the update
telemetry read the settled state and need no change.

### Decision B: Not the leaky substrate

`dynamics: leaky` already couples ohmically, through the same kind of operator, but carries membrane
potentials across steps; it learned the thermal cell at 20% against the settling map's 64% (Logbook
080), mostly from the carried state. This change keeps the settling map memoryless, so the corrected
model differs from every registered reading in the gap coupling's form alone.

### Decision C: The selection rule, fixed before the pilot runs

**The pilot** (seeds 9871–9886, disjoint from every band) runs on both block-V cells on Emmons 2024, the
thermal cell at target 35 and hard350, 3,000 episodes. On each cell: for each candidate model (`ohmic` at
the defaults; `log_count`), the wild type and the gap-only null, each learning and frozen; the
**additive-count** wild type learning, the substrate every registered reading used, read only as
description; and the MLP-PPO control, learning and frozen (its frozen floors derived from the committed
learning configs). 2 cells × 11 arms × 16 seeds = **352 runs**.

1. **The cells must be readable**: the MLP control beats its frozen floor on both cells (the paired 80%
   interval of learning − frozen plateau above zero). If not, the pilot is unreadable and nothing is
   chosen.
2. **A model is eligible on a cell** when its wild type and its gap-only null each beat their own frozen
   floors and the two do not both plateau at or above 90%. This is the readability gate the model's
   point-worm contrast will need, and the only place the null enters the rule: a model on which either
   wiring cannot learn could not carry that contrast.
3. **Ohmic is chosen** if it is eligible on both cells, unless on either cell its wild type is **shown
   slower** than log's by more than that cell's margin δ: the paired-by-seed 80% bootstrap interval of
   `auc_success(ohmic) − auc_success(log)` (the instrument's own `paired_seed_wilcoxon_bootstrap`) lies
   entirely below −δ. The margins are Logbook 080's for the same question, a new substrate's wild type
   against the settling one's: **δ = 0.0239** on the thermal cell (Logbook 078's registered minimum
   there) and **δ = 0.0143** on hard350 (two-thirds of A.6's committed lead over the chemical-only null).
4. Otherwise **log-compressed counts** are chosen if eligible on both cells.
5. Otherwise **neither**: the result is recorded and the substrate decision returns to the maintainer.

**Operating characteristic**, from Logbook 080's paired spreads (sd 0.104 thermal, 0.132 hard350) at 16
seeds: if the two models' wild types learn equally fast, ohmic is chosen with probability about 0.95; if
ohmic is slower by 0.025, 0.76; by 0.05, 0.38; by 0.10, 0.01. The rule prefers ohmic, the maintainer's
stated preference on fidelity, unless the evidence says it is the worse instrument. At 8 seeds the same
rule would choose an ohmic model 0.05 slower with probability 0.56, which is why the pilot runs 16.

**What the rule never reads**: the gap between the wild type and the null, on either model. On log
counts that gap is already known as description (+0.012 on 16 seeds, G.0(e)); on ohmic it is unknown
and stays unread until the registered contrast. The rule reads floors, saturation and the wild type's
learning speed only, so the choice of model is not a choice of answer.

### Decision D: The operating point, measured before the pilot, and the ohmic scale calibrated

At no run cost, on untrained brains with `sign_scale_calibration.py`'s 64 fixed inputs and the pilot's
seeds, the saturated-unit fraction and the action mean's spread across the inputs (the sensor-to-action
gain), on both cells.

*As first registered*, the pilot launched only if ohmic's spread was at least a quarter of the additive
count model's. **The probe stopped it** ([probe.json](../../../docs/experiments/logbooks/supporting/097-gap-substrate/probe.json)):
at the built gap matrix's scale ohmic's spread is 0.0003 against the additive model's 0.24, on both cells.
The probe also showed the yardstick was the artefact: every brain G.0 showed learns as well as the count
model falls far below a quarter of its spread (log counts 0.035, unit weights 0.026, gap junctions off
0.021, against plateaus of 67–71%), because the additive count model's large spread is its gap
amplification.

*Amended 2026-10-17 by the maintainer, before any pilot run*: the reference is the **gap-off brain**, a
known learner (67% in G.0), and the ohmic conductance scale is **calibrated** on the same untrained
brains, as Logbook 091 calibrated a weight scale: on the grid ×1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001,
the largest scale whose spread is at least half the gap-off brain's on both cells. **It is ×0.03**
(spread 0.0162 against a floor of 0.0125 on both cells; ×0.1 gives 0.0078). At ×0.03 the median neuron
keeps 93% of its own drive and ALA, the most strongly coupled, 29%: strong pairs still couple, weak ones
barely do. The pilot's ohmic arms run at that scale (`gap_weight_scale: 0.03`); the log arms are as
registered. If no scale on the grid had passed, the pilot would not have launched.

### Decision E: Byte identity and the tests

`gap_coupling: additive` is the default and computes exactly what it computes today: the settling paths
are unchanged on that branch, and pinned digests (gap matrix, weights, the next draw, as G.0's tests pin
them), the state-dict keys and an output comparison hold it. Tests on `ohmic`: the operator against a hand
computation, its rows summing to 1; **the no-gain test**, a pair at conductance 100 with no chemical edges,
started saturated with no drive, holds under `additive` and falls to zero under `ohmic` in one iteration,
and with a small drive on one neuron saturates under `additive` and stays near the drive under `ohmic`;
the sensors relayed through `M⁻¹`; the modulator field's held current passed through `M⁻¹`; the single and
batched paths agree; every refusal fires.

### Decision F: What follows, each under its own registration

- **G.0-A's point-worm contrast** (seeds 3901–3964): the wild type against the gap-only null on the
  thermal cell at target 35, on the chosen model, learning and frozen, 64 seeds, its minimum **0.0577**.
  It re-establishes the effect on the corrected model, alone (protocol principle 6): the outlier nulls and
  any other ablation register only after it reads. Gap-off is the same brain under both couplings and is
  G.0(d)'s committed level; it is not re-run.
- **D24 under ohmic**, defined before that contrast registers: the matched-statistics table in effective
  units (the entries of `M⁻¹`), and a shortcut audit that counts a gap component as reached at once, since
  under ohmic coupling drive crosses a whole component within one iteration.
- **092's body contrast on the corrected model**, after M.1 reads out (D25), seeds fixed then. This
  replaces "before anything else" in the tracker's G.0-A item and the roadmap's G.0 entry, as a dated
  amendment recording the maintainer's 2026-10-17 decision.
- **The decomposition restated** (Logbooks 074, 075, 078, 079, 082, 092), once both have read.
- G.1–G.3b then register on the corrected model, or are deferred with a destination.

**The minimum, a dated decision.** 0.0577 is Logbook 082's registered minimum on this cell, two-thirds of
Logbook 078's +0.0865. D23 re-fixed the Emmons minimum at 0.0894 from G.0(a)'s anchor, but that anchor
measured the count-weighted model whose lead G.0 found to be an artefact, so it does not size an effect
on the corrected model. The maintainer chose 0.0577 on 2026-10-17; at 64 seeds the detectable effect is
about 0.033–0.042.

## Risks / Trade-offs

- **Ohmic coupling may change the operating point** enough to stop learning, as 080's first leaky
  registration did. Decision D measures it before any run.
- **Ohmic coupling may learn slowly.** Then the rule picks log-compressed counts, and the fidelity question
  is recorded as open rather than forced.
- **Ohmic coupling may abolish any gap effect** on the corrected model. That is a legitimate reading of the
  next contrast, not a reason to choose the other model; the selection rule cannot see it.
- **The plastic rules, node perturbation, activity traces, e-prop and `plastic_gaps` are refused under
  ohmic**; extending them is a later change if a block needs it.
