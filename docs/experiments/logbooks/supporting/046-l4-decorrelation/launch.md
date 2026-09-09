# Decorrelation test launch record

Written and committed before the pilot ran, per the registration.

- **Date**: 2026-09-09
- **Pinned state**: `feat/l4-decorrelation` at the commit implementing the two terms, the four
  arm configs and the harness. The commit carrying this record precedes the pilot.
- **The prediction under test**: the sign-grounding test found that grounding the substrate's
  synapse signs made reward-free Hebbian learning substantially worse (wild-type 31.5 → 14.0,
  rewired null 17.4 → 9.1) and predicted that a rule with an anti-Hebbian or decorrelating term
  would recover the loss.
- **The two variants**: `anti_hebbian_inhibitory` negates the Hebbian term at synapses the atlas
  grounds as inhibitory — **no hyperparameter**, so it cannot be tuned into a result — and `oja`
  subtracts `η · γ · y² · w`, needing no transmitter identity.
- **Protocol**: the sign-grounding test's own Hebbian protocol, unchanged except for the rule
  keys. Four arms (`wt_antihebb`, `rn_antihebb`, `wt_oja`, `rn_oja`), seeds 1–16 paired, 1000
  episodes, plateau-tail full-clear success. The single registered extension is a fresh run at
  1.5× replacing a shorter log the plateau detector marks non-converged. 64 runs.
- **Comparators**: the sign-grounding test's committed per-seed table
  (`supporting/044-l4-atlas-signs/per-seed.csv`) on the same seeds, paired seed for seed. The
  grounded Hebbian arms are **not** re-run. Panel 2's random-sign means (31.5, 17.4) are the
  `full_recovery` target and are descriptive.
- **Family and verdict**: fixed in `scripts/analysis/l4_decorrelation.py` — D1 wild-type
  anti-Hebbian over the committed wild-type grounded Hebbian, D2 the same for Oja, D3 and D4 the
  wiring contrast under each variant, corrected together under BH-FDR at α = 0.05. Verdict in
  order: `insufficient_seeds`, **`no_recovery`** (neither D1 nor D2 confirms — the outcome in
  which the prediction fails), `recovery_specific`, `recovery_general`, `recovery_both`. D3 and
  D4 annotate and never decide.
- **Annotations**: `full_recovery` (the 80% bootstrap interval of an arm's mean plateau tail over
  seeds 1–16 reaching the committed random-sign mean for its wiring) and the mean decorrelation
  share from each run's own telemetry, so a recovery whose share is near zero is recorded as
  attributable to something other than the term.

## The pilot, declared before it ran

Only the Oja coefficient has no value to inherit. Seeds 1–2 of the wild-type grounded arm at the
same 1000-episode budget over `γ ∈ {0.01, 0.1, 1.0}`; pin the highest mean plateau tail, ties to
the smaller `γ`. 6 runs.

The **anti-Hebbian variant has no pilot and no pin**: it is the existing update with one factor of
−1 on the subset the atlas identifies, so there is nothing to choose. That is deliberate — the
variant carrying the primary test cannot be tuned toward a result.

## Commands (from the repository root)

```bash
# Pilot (Oja coefficient, seeds 1-2, 1000 episodes)
uv run python scripts/run_campaign.py --config <each pilot config> \
  --seeds 1-2 --runs 1000 --output-dir campaigns/l4-decorrelation-pilot -- --theme headless --track-experiment

# Test (seeds 1-16, 1000 episodes)
P=configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic_hebbian
uv run python scripts/run_campaign.py \
  --config ${P}_atlassigns_antihebb.yml --config ${P}_rewired_null_atlassigns_antihebb.yml \
  --config ${P}_atlassigns_oja.yml --config ${P}_rewired_null_atlassigns_oja.yml \
  --seeds 1-16 --runs 1000 --output-dir campaigns/l4-decorrelation -- --theme headless --track-experiment
```

## Analysis

`uv run python scripts/analysis/l4_decorrelation.py --campaign-dir campaigns/l4-decorrelation --out docs/experiments/logbooks/supporting/046-l4-decorrelation/panel.json --csv docs/experiments/logbooks/supporting/046-l4-decorrelation/per-seed.csv --curves docs/experiments/logbooks/supporting/046-l4-decorrelation/curves.csv`

## Pilot results

Run 2026-09-09, 6 runs, all exit 0, 8 minutes on 16 workers. Plateau-tail full-clear success on
the two pilot seeds; the committed grounded Hebbian arm scores 10.0 and 32.8 on these seeds
(mean **21.4**).

| `γ` | seed 1 | seed 2 | mean |
|---|---|---|---|
| 0.01 | 10.0 | 27.6 | 18.80 |
| **0.1** | 12.4 | 32.0 | **22.20** |
| 1.0 | 10.8 | 28.0 | 19.40 |

**Pin**: `γ = 0.1`, by the declared criterion (highest mean; no tie), which is the value the
committed arm config already carried.

Read before the test and recorded here so the test is read against it: on these two seeds the Oja
term is within a point of the comparator rather than above it, and both seeds are dominated by
which seed they are (10.0 against 32.8 in the comparator itself). Two seeds license nothing; this
is a pin, not a signal.

**The anti-Hebbian variant has no pilot**, as declared — there is no coefficient to choose.

## Test results

Run 2026-09-09, 64 runs, all exit 0, 42 minutes on 16 workers, no extensions needed.
**Verdict `no_recovery`**: D1 −0.90 (q = .487), D2 −2.60 (q = .487); the wiring contrast is
unconfirmed under both variants (D3 +2.58, D4 +2.02) and `full_recovery` is false everywhere.

The decorrelation share separates the two failures and is why it was registered: the anti-Hebbian
arm's 0.057 is exactly the grounded inhibitory fraction of the substrate (214 of 3,709 synapses),
so the term redirected every synapse it could and changed the outcome by −0.9 points; the Oja
arm's 0.022 says its term contributed a little over two percent of the update at the pinned
`γ = 0.1`, a weak test of the idea — the grid's top value of `1.0` would have contributed about
18%, and the pilot chose against it on two seeds. Full reading in `details.md`. The share figures
are the corrected ones: the first computation compared quantities in different units and the 64
runs were repeated under corrected telemetry, reproducing every per-seed value exactly.
