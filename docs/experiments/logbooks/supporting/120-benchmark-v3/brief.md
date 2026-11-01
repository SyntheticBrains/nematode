# Benchmark v3: the planning brief's readings (2026-11-01; description)

A research pass before planning read these real data sets with the benchmark's own detector. They count as seen before Part A.

## Held-out data read

- **López-Cruz 2019:** per-batch levels and log early/late for all 27 wild-type batches, including the 10 held out (`lopezcruz2019_split.json`). Held-out log early/late against the 17 band batches is within |z| ≤ 1.14.
- **Roberts 2016:** detector onset rates for all five cohorts, including the held-out C186 and C21:
  - 3.24 per minute pooled (2.93–3.42 by cohort);
  - the tracker's reverse state is 17.3% of tracked frames.
- **Calhoun 2014:** detector rates for all 47 worms, about 1.78 per minute pooled. Per 2-min bin, the rate peaks at 5.9 around 6–8 min.
- **Chen 2025:** detector onset rates for all 8 recordings: 1.74, 0.80, 1.62, 1.14, 0.72 and 0.68 per minute on the standard protocol, and 1.50 and 1.36 off it.

## What it established

- **On real tracks, our detector counts 2–3× López-Cruz's reorientations** (Calhoun 2.2× early, 3.0× late; Roberts about 2.4–2.9×, inferred). So a level against López-Cruz's counts would need a cross-lab factor.
- **An early-window level would not catch restart 2** (3.8–5.2 per minute early, against Roberts' 3.2).
- **v2's linear early/late band** (4.21 ± 1.62) accepts −0.66 to 9.09. On a log scale, restart 2 fails while every held-out batch passes.
- **v2's on-lawn reversal rate is reported, not graded:** no open-loop point worm reached it through the tracker (1.31 against 2.28), and the off-food detector's recall on food is 0.43.

## Records

- `probe.json` and `probe.py.txt` (the script as run): the discriminability probe (animals at 99,940,000).
- `preflight.json` and `preflight.py.txt` (the script as run): the adversary and oracle preflights (99,950,000 and 99,960,000, CMA 9423).
