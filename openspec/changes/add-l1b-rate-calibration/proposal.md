# Does L.1's interaction survive the learning rate? (L.1b)

## Why

[L.1](../../../docs/experiments/logbooks/066-l4-readout-width.md) read `pooling_hid_structure`: at
`plasticity_rate` **0.001**, widening the readout from the four-class pool to one weight per motor
neuron helps the wild type and hurts its degree-preserving null — interaction **+0.2818** on
`auc_success` (q = 0.000, 74/96), the wild type ahead at the per-neuron width by +0.1852 where at the
pooled width the null led by 0.0966. It is the phase's only positive for the wild-type wiring under a
biologically plausible learner.

[L.4/L.5](../../../docs/experiments/logbooks/067-l4-feature-ablations.md) then measured, as a
by-product of its rate-matched baseline, the two wide learning arms at **0.0001** on the same 96
seeds: the wiring effect there is **−0.0977, null ahead**. Both wirings learn better at the lower
rate, the null by +0.44 and the wild type by +0.16. L.1's headline form — *the sign flips* — does not
hold one decade below the rate it ran at.

That rate was inherited, not chosen for the width. R.2 pinned 0.001 on the 8-parameter readout and
waived its registered check; [L.0](../../../docs/experiments/logbooks/064-l4-frozen-features.md) ran
the check at the pooled width on the wild type alone and found 0.001 the optimum there (seeds
101–104: 18.085 foods against 14.488 at 0.0001 on the plateau-tail mean every contrast here reads;
L.0's own table published the whole-run mean, 17.417 against 12.086, and the ordering is the same on
either); L.1 carried it to a 78-parameter readout and to the null without a sweep. Protocol principle 7 — a setting pinned at one width is a hypothesis at another —
has already been paid for once in this programme.

What is **not** known is the thing the synthesis has to say: whether L.1's **registered primary, the
interaction**, is a property of the wiring or of the rate. At 0.0001 the two wide cells exist; the two
pooled cells do not. Two arms close it.

> *At the matched rate 0.0001, does widening the readout still help the wild type more than the
> shuffle — or was L.1's interaction a property of 0.001?*

This runs **before Z.1** because principle 12 says re-read before shipping, and the one re-read that
can change what the synthesis says about its own headline costs 192 runs.

## What changes

- **Two new learning arms** at seeds **1–96**: the pooled readout at 0.0001, wild type and null
  (`..._hard350_eprop_readout_only_r1e4{,_rewired_null}.yml`), each one key from its committed L.0/L.1
  pooled parent. The wild-type config exists from L.0's rate check and is promoted to a registered
  arm with its header rewritten; the null's is new. **192 runs.** Every other cell is committed:

  | cell | rate | source |
  |---|---|---|
  | `wt_pooled`, `rn_pooled` | 0.0001 | **this change** |
  | `wt_wide`, `rn_wide` | 0.0001 | L.4's rate-matched baseline, `campaigns/feature-ablations` |
  | the four floors | inert | L.1, `campaigns/readout-width` |
  | L.1's four learning arms | 0.001 | L.1, for the three-way contrast only |

- **The primary is the interaction at 0.0001**, `(wt_wide − wt_pooled) − (rn_wide − rn_pooled)`, per
  seed, L.1's registered primary re-read at one rate. Both main effects at 0.0001 beside it, four
  learning gates (each 0.0001 arm against its own-width floor) read first, and one secondary the
  question makes unavoidable: the **three-way**, L.1's interaction at 0.001 minus this one at 0.0001,
  per seed, which is the size of the rate dependence itself. **Eight tests, one BH-FDR family.** The
  pooled and wide priors are L.1's committed tests (q = 0.462) and are not re-tested; the floors have
  not changed.

- **A minimum effect as a decision rule, carried from L.1 as a fraction**: `survives` requires the
  interaction at 0.0001 to be significantly positive **and** `abs(Δ) ≥ 0.141` — **half** of L.1's
  +0.2818. The pool-hiding claim survives only if at least half of it is still there at the matched
  rate; a significant interaction below that is *shrunk below half* and reads rate-specific. Sized in
  the design from the cells' realised spreads: ~98% power at the minimum if the pooled cells spread as
  they did at 0.001, ~87% if they spread wider.

- **The wide cells at 0.0001 are reused under the byte-identity requirement**, and so are L.1's
  floors and learning arms — one seed per reused arm re-run under the current path before any
  registered seed, eight runs, compared on every parsed field. Any field differing re-runs that arm's
  cell in full.

- **A harness** `scripts/analysis/l4_rate_calibration.py`, the L.1/L.4 sibling pattern: three
  campaign directories in, `connectome_structure_efficiency.analyse` called once per (rate, width)
  pair and unmodified, L.1's helpers imported. **L.1's, L.4's and block V's harnesses are read-only,
  asserted.**

- **What the reading conditions, and what it does not change.** L.1's verdict stands as read at
  0.001, whatever this returns; a committed verdict is not rewritten. What this adds is the
  operating-point condition the synthesis carries beside it: *the pool hid structure at every rate
  tested* or *at 0.001 only*. It does not decide which rate is "right" — both wirings learn better at
  0.0001, so the pinned rate is the worse operating point for either — and it does not ablate
  anything against the 0.0001 baseline; that is what L.4 did prematurely, and the rung that would is
  the phase after 7's.

## The registered readings

| reading | when |
|---|---|
| `pool_effect_survives_the_rate` | interaction at 0.0001 significantly **positive** and abs(Δ) ≥ 0.141. Widening still favours the wild type relative to the null at the matched rate, with the wide wiring effect itself negative there (known: −0.0977) — so the pool-hiding claim survives the rate and the sign-flip form does not; the record says both |
| `pool_effect_is_rate_specific` | no significant interaction — **a failure to detect**, size and CI carried — **or** significant positive but below 0.141, named *shrunk below half*. L.1's positive is a property of 0.001 |
| `width_favours_the_shuffle_at_this_rate` | interaction significantly **negative**. Reported as the reverse direction and not explained |
| `no_learning` | a gate fails on a 0.0001 arm |
| `insufficient_seeds` | fewer than 5 paired seeds survive, where 2⁻ⁿ first clears 0.05 |

## Impact

- Affected specs: `plasticity-evaluation`
- Affected code: `configs/scenarios/foraging/` (one new config, one promoted with its header
  rewritten), `scripts/analysis/l4_rate_calibration.py` (new)
- **No package code changes.**
- **`l4_readout_width.py`, `l4_feature_ablations.py`, `connectome_structure_efficiency.py`,
  `wiring_premise.py` are READ-ONLY.**
