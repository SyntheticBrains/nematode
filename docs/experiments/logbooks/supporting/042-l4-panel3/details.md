# Panel 3 details

Analysis by `scripts/analysis/l4_panel3.py` from `campaigns/l4-panel3` (the manifest beside this
file lists every log; no run needed the registered extension), with the frozen floors for seeds
17–64 and the descriptive pooling of seeds 1–16 read from panel 2's committed table
`supporting/041-l4-panel2/per-seed.csv`. `panel3.json` is the full output; `per-seed.csv` and
`curves.csv` are the per-run table and learning curves.

## Verdict: `inconclusive`

R1's interval is clear of zero and its corrected q is above α; the map assigns neither
`specific_wiring` nor `degree_statistics`. R2 does not reach α either.

| test | statistic | value | p | q | result |
|---|---|---|---|---|---|
| R1 wt vs rn Hebbian, seeds 17–64 | mean Δ, 80% CI, +seeds | +8.1, [+2.3, +14.2], 24/48 | .188 | .188 | fail |
| R2 competent-fraction discordance, seeds 17–64 | b wild-type-only / c rewired-only / both | 11 / 6 / 5 | .166 | .188 | fail |

Both tests are complete (every registered seed present).

## Per-arm plateau tails, seeds 17–64

| arm | mean | median | q75 | max | competent (≥ 20%) |
|---|---|---|---|---|---|
| wt_hebbian | 19.9 | 4.2 | 34.2 | 86.0 | 16/48 (0.33) |
| rn_hebbian | 11.9 | 2.8 | 15.7 | 56.0 | 11/48 (0.23) |

Sorted wild-type: 0×8, 1×7, 2×5, 3, 4×5, 5, 6, 6, 11, 14, 17, 20, 25, 26, 34, 36, 36, 42, 46, 56,
65, 70, 76, 78, 79, 84, 86. Sorted rewired: 0×10, 1×3, 2×10, 3×4, 4×3, 6, 6, 7, 12, 13, 16, 16, 22,
28, 30, 33, 38, 40, 41, 53, 53, 55, 56.

Paired deltas: 24 positive, 22 negative, 2 ties; spread 32.2. Three seeds are dead (< 1%) on
both wirings.

Learning gains (Hebbian minus own frozen floor): wild-type +6.9 (21/48) on 17–64, +11.6 (32/64)
pooled; rewired +3.9 (29/48) on 17–64, +5.3 (38/64) pooled.

## Pooled 1–64 (descriptive)

Mean Δ +9.6, 80% CI [+4.3, +15.0], 34/64 positive; discordance 19 wild-type-only against 10
rewired-only; competent fractions 0.39 against 0.25.

## Reading

- **The estimate shrank with each fresh look.** +16.2 on seeds 1–8 (panel 1's runs), +11.9 on
  seeds 9–16, +8.1 on seeds 17–64. The first, descriptive signal was the most favourable draw;
  the pooled 64-seed estimate is +9.6 with an interval clear of zero, and the sign count is at
  chance (24/48 here, 34/64 pooled).
- **The wiring's mark is in the level of the good fixed points, not their frequency.** The
  wild-type's eight best seeds sit at 56–86%; the rewired-null's best eight at 33–56%. Alignment
  finds a competent fixed point on 16 of 48 wild-type seeds against 11 of 48 rewired, a
  difference R2 cannot confirm (p = .17). A paired rank test sees a 50/50 sign split and reports
  no shift; the mean is carried by a handful of large positive deltas from that upper tail.
- **Power, as registered.** The design gave R1 roughly 63–75% power for a +12 to +14 effect at
  the corrected threshold. At the observed +8 (d ≈ 0.25) it was nearer 50%. A null here was a
  likely outcome even for a real effect of this size, and the registration forbids extending
  further on the same hypothesis.

## Campaign facts

- 96 runs (2 arms × seeds 17–64 × 1000 episodes), 20:06–21:08 on 16 workers.
- No run needed the registered extension; no tracebacks; the runner exited 0.
- Floors: panel 2's sweep values for seeds 17–64, read from the committed table.
