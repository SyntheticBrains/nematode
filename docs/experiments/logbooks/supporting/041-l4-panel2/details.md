# Panel 2 details

Analysis by `scripts/analysis/l4_panel2.py` from the manifest beside this file (Hebbian panel with
its three registered extensions) and the prior-sweep campaign directory. `panel2.json` is the full
output; `per-seed.csv` and `curves.csv` are the per-run table and learning curves.

## Verdict: `inconclusive`

The primary P1 (wild-type Hebbian over rewired-null Hebbian, degree-scaled initialisation, seeds
1–16) has a positive interval clear of zero but is not significant after correction, so the map
assigns neither `specific_wiring` nor `degree_statistics`.

| test | mean Δ | 80% CI | p | q | +seeds | outcome |
|---|---|---|---|---|---|---|
| P1 wt vs rn Hebbian, degree init | +14.1 | [+2.7, +25.0] | .074 | .277 | 10/16 | fail |
| P2 wt vs rn Hebbian, count init | +2.0 | [−7.6, +11.8] | .350 | .467 | 8/16 | fail |
| P3 count vs degree, wt Hebbian | −15.6 | [−27.1, −2.5] | .920 | .920 | 6/16 | **reverse** |
| P4 wt vs rn frozen, seeds 1–64 | +3.3 | [−0.0, +6.8] | .139 | .277 | 36/64 | fail |

Annotations: count-scaled initialisation does not preserve the contrast (P2), does not improve
the wild-type fixed point (P3 reverses), and the prior does not differ at the registered level
(P4).

## Per-arm plateau tails

| arm | n | mean | sorted per-seed values | competent (≥ 20%) |
|---|---|---|---|---|
| wt_hebbian | 16 | 31.8 | 0 0 0 0 3 6 16 24 26 31 41 68 69 70 71 80 | 9/16 |
| rn_hebbian | 16 | 17.4 | 0 0 0 2 6 6 7 14 17 18 18 24 31 43 46 48 | 5/16 |
| wt_hebbian_count | 16 | 15.6 | 0 0 0 0 0 1 1 2 3 8 9 14 19 49 67 81 | 3/16 |
| rn_hebbian_count | 16 | 14.2 | 0 0 0 0 0 0 1 1 2 2 3 30 36 43 49 56 | 5/16 |

Learning gains (Hebbian minus own frozen, seeds 1–16): wt_hebbian +25.7 (11/16), rn_hebbian +9.6
(9/16), wt_hebbian_count +9.3 (7/16), rn_hebbian_count +9.2 (8/16).

## Prior sweep (frozen arms, seeds 1–64, 600 episodes)

| arm | mean | q25 / median / q75 | max | competent fraction |
|---|---|---|---|---|
| wt_frozen | 11.3 | 1.3 / 3.7 / 14.2 | 74.0 | 0.22 |
| rn_frozen | 8.0 | 0.7 / 2.3 / 10.8 | 43.3 | 0.16 |
| wt_frozen_count | 10.9 | 2.7 / 5.3 / 13.0 | 52.0 | 0.16 |
| rn_frozen_count | 7.7 | 0.5 / 2.7 / 8.3 | 52.0 | 0.11 |

Descriptive frozen pairs: count minus degree on the wild-type −0.4 (CI [−2.5, +1.6], 32/64), on
the rewired −0.3 (CI [−2.0, +1.5], 26/64). The count structure leaves the untrained prior where
it was.

## Reading

- **The primary held its size and lost its test.** Seeds 1–8 are panel 1's runs (their first
  1000 episodes reproduce panel 1's logs exactly, all four degree-scaled arms) and give +16.2 on
  5/8; the new seeds 9–16 give +11.9 on 5/8. The effect did not shrink on fresh seeds, but with
  per-seed deltas whose spread is 36.5 points (six seeds between −15 and −46 against ten
  between +6 and +70) a one-sided Wilcoxon at n = 16 reaches p = .074, and the family
  correction leaves q = .277. The outcome is bimodal on both arms; the wild-type has more seeds
  in the upper mode (9 competent against 5) and a higher upper mode (68–80 against 43–48).
- **Where the advantage lives.** The untrained prior differs by +3.3 (P4, interval touching
  zero; competent 0.22 against 0.16, and the heaviest upper tail — 74, 49, 47, 44 — is the
  wild-type's). After Hebbian alignment the difference is +14.1, and the wild-type's learning
  gain is +25.7 against the rewired null's +9.6. Descriptively, most of the wild-type's
  advantage is created by the reward-free Hebbian fixed point rather than present in the prior;
  the registered tests confirm neither.
- **Count-scaled initialisation hurts the wild-type Hebbian fixed point and erases the
  contrast.** P3 reverses (−15.6, interval entirely below zero): the wild-type Hebbian arm
  falls from 31.8 to 15.6 mean and from 9 to 3 competent seeds, its learning gain from +25.7 to
  +9.3, while the rewired arm is unchanged (17.4 → 14.2) and the frozen priors are unchanged
  under both wirings. Linear count scaling concentrates each neuron's input on its few
  high-count edges (the largest input edge carries a median 29% of a neuron's input); the
  Hebbian rule then aligns onto an input already dominated by those edges, and on the wild-type
  wiring that dominant structure is not the one the reward-free fixed point exploited under
  equal-magnitude initialisation. The count structure is anatomy the untrained network does not
  use (P4-level priors unchanged) and that this local rule uses badly.
- **Three seeds needed the registered extension** (wt_hebbian 14, wt_hebbian_count 3,
  rn_hebbian_count 10; no plateau at 1000). All three converged at 1500 with tails within
  four points of their 1000-episode values; the family and verdict were unchanged by them.

## Campaign facts

- Sweep: 256 runs (4 frozen arms × seeds 1–64 × 600 episodes), 08:38–10:02 on 16 workers.
- Hebbian panel: 64 runs (4 arms × seeds 1–16 × 1000 episodes), 10:02–10:49.
- Extensions: 3 runs at 1500 episodes, 10:49–10:59.
- Reproduction check: the degree-scaled frozen arms' 600-episode logs and the degree-scaled
  Hebbian arms' first 1000 episode lines are identical to panel 1's logs on seeds 1–8 (32 of 32
  comparisons).
- No tracebacks in any log; both runners exited 0.
