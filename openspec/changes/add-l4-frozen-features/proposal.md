# Is the wiring legible to a learner that reads it? (L.0)

## Why

Phase 7's flagship was a 2×2 — learning rule × wiring — asking whether the wild-type connectome
becomes load-bearing under a rule the animal could host. For rules that **write** the wiring, that is
answered, and the answer is no twice over.
[R.1c](../../../docs/experiments/logbooks/061-l4-reduced-perturbation.md) took the perturbation from
1208 draws per decision down to 39 and returned `not_reducible`;
[R.1d](../../../docs/experiments/logbooks/062-l4-frozen-readout.md) showed the frozen readout is part
of the limit without any arm reaching competence; and
[R.2](../../../docs/experiments/logbooks/063-l4-eprop.md) closed the family —
**e-prop reaches competence on this cell, and the arm that does so has its chemical matrix frozen.
Every arm that writes the wiring does worse, by 3.9 to 15.9 foods.**

That leaves one form of the question unasked, and it is the one D14 restates the claim around:
**is the wiring legible to a learner that *reads* it rather than writes it?** `readout_only` — an
8-parameter linear map over four pooled motor-class means, learned by its own exact gradient on
**frozen** recurrent weights — is the first plausible learner on this substrate to reach competence
(**17.570 foods of 20, 52.61% full clear, 16/16 seeds**). It is the opposite of a wiring-indifferent
learner: the frozen wiring is the only thing it has.

So this runs the 2×2's primary contrast under it: **wild type against its degree-preserving rewired
null, both frozen, with the readout learning on top.**

## What changes

- **Four arms on the `hard350` cell**, which is where the three things this needs already meet:

  | arm | wiring | learner | runs |
  |---|---|---|---|
  | `wt_readout_only` | wild type | readout learns, `w_chem` frozen | 32 |
  | `rn_readout_only` | degree-preserving rewired null | the same | 32 |
  | `wt_frozen` | wild type | nothing learns | 32 |
  | `rn_frozen` | rewired null | nothing learns | 32 |

  The learning arms differ from the committed `..._eprop_readout_only.yml` by the `wiring` key
  **alone**, and the floors from `..._eprop_frozen.yml` likewise — asserted by an exact-key test, as
  every arm in this programme has been.

- **The metric and the bar are block V's, unchanged**: `episodes_to_30pct_success` through the
  committed `connectome_structure_efficiency.py`, against the registered **≥ 20%** minimum on
  time-to-competence, with the two learning gates (each wiring against its own frozen floor) and the
  untrained-prior check block V ran.

- **32 paired seeds, matching V.3's power rather than R.2's count.** This is the load-bearing
  protocol choice and the arithmetic is registered here because a null closes the phase. At 16 pairs
  a one-sided sign test needs **12/16 (75%)** positive to reach p ≤ 0.05, and V.3's observed per-seed
  win rate on this exact contrast was **21–26 of 32, or 65.6–81.3%** — giving 16 seeds **30.7–83.4%**
  power across the comparator's own range, and **57.3%** at its midpoint of 73.4%. Thirty-two
  pairs give **43.4–97.3%**, and **79.2%** at that midpoint. The registered test is a paired *rank* test under BH-FDR,
  which uses magnitudes and so has somewhat more power than the sign test; these figures are a
  conservative floor and are stated as such.

- **The direct comparator is a committed result on the same cell.** V.3 ran this contrast on
  `hard350` under **PPO** over 32 paired seeds and found the wild type competent in **892 episodes
  against the null's 1165 — +23.5%**, above the 20% bar, three of four efficiency metrics significant
  (q = 0.003, 0.017, 0.029), both learning gates 32/32, the untrained prior indistinguishable
  (−0.01, q = 0.841). So this is not a contrast run somewhere new: it asks whether **the wiring
  advantage PPO shows on this cell survives a plausible learner that reads the wiring instead of
  rewriting it**, at the same seed count, the same metric and the same bar.

- **What a positive result may and may not be cited as, registered before the run.** A positive is a
  **performance claim** under the phase's claim discipline: the wild-type wiring supplies better
  fixed features to a small learned readout than a degree-matched shuffle does. It is **not** a
  dynamics claim, it does **not** satisfy D2's primary (which requires *plastic* wild-type to beat
  *plastic* rewired-null, and nothing here makes the wiring plastic), and it cannot convert Phase 7's
  SPLIT into a GO.

Out of scope: V.1's thermal cell, whose operating point was never calibrated for this learner
(R.2's σ, action noise and rate were calibrated on `hard350`); any change to the readout's width,
which is **L.1** and is gated on this result; and the feature ablations L.4 and L.5, gated the other
way.

## Capabilities

**Modified**: `plasticity-evaluation` — a wiring contrast run under a learner that does not write the
wiring states which of the substrate and the learner the result is about; and a campaign whose null
outcome carries a registered consequence states its power against the comparator it will be read
beside, before it runs.

## Impact

- New: two configs (the rewired-null counterparts); `scripts/analysis/l4_frozen_features.py`; records
  under `supporting/064-l4-frozen-features/`; Logbook 064.
- Edited: the experiments index, `CHANGELOG.md`, the tracker (L.0), the roadmap only if the reading
  changes.
- Compute: **128 runs** — four arms × 32 seeds at ~1800 s — about **5h** at the measured parallelism,
  plus a 4-run pilot on disjoint seeds and the rate check.
