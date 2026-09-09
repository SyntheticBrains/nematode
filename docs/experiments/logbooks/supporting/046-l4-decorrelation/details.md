# Decorrelation test details

Analysis by `scripts/analysis/l4_decorrelation.py` over the 64 runs in
`campaigns/l4-decorrelation/logs` (4 arms × seeds 1–16 × 1000 episodes, all exit 0, 42 minutes on
16 workers). **No run needed the registered extension.** `panel.json` is the full output;
`per-seed.csv` and `curves.csv` are the per-run table and learning curves. The comparators are the
sign-grounding test's committed per-seed values on the same seeds; no grounded Hebbian arm was
re-run.

## Verdict: `no_recovery`

Neither decorrelating term recovers what grounding the signs cost. This is the registered outcome
in which the sign-grounding test's prediction fails.

| test | contrast | mean Δ | 80% CI | q | +seeds | result |
|---|---|---|---|---|---|---|
| D1 | wt anti-Hebbian − wt grounded Hebbian | −0.90 | −5.48 … +3.00 | 0.487 | 9/16 | fail |
| D2 | wt Oja − wt grounded Hebbian | −2.60 | −5.80 … +0.45 | 0.487 | 3/16 | fail |
| D3 | wt − rn under anti-Hebbian | +2.58 | −5.05 … +9.78 | 0.487 | 8/16 | fail |
| D4 | wt − rn under Oja | +2.02 | −4.53 … +8.26 | 0.487 | 7/16 | fail |

| arm | mean | 80% CI | random-sign target | reaches it | decorrelation share |
|---|---|---|---|---|---|
| wt_antihebb | 13.1 | 7.4 … 19.0 | 31.5 | no | **0.057** |
| rn_antihebb | 10.5 | 6.6 … 14.8 | 17.4 | no | 0.058 |
| wt_oja | 11.4 | 6.7 … 16.6 | 31.5 | no | **0.0026** |
| rn_oja | 9.3 | 6.0 … 13.0 | 17.4 | no | 0.0026 |

`full_recovery` is false for every arm; no interval comes near its random-sign target.

## The two failures are not the same failure

The decorrelation share is what separates them, and it is why the annotation was registered.

- **The anti-Hebbian variant did everything it could and it was not enough.** Its share is 0.057,
  which is exactly the grounded inhibitory fraction of the substrate: the atlas grounds **214
  inhibitory synapses out of 3,709**, or 5.8%. The term redirected every one of them and left the
  rest alone, as specified. So D1's failure is a real answer about the variant *as it can be built
  at this fidelity*: negating the update on every inhibitory synapse the transmitter atlas can
  identify changes the outcome by −0.9 points. **There is not enough grounded inhibition here to
  build a brake out of.**
- **The Oja arm did not test its term.** Its share is 0.0026 — the term contributed a quarter of a
  percent of the update's magnitude at the pinned `γ = 0.1`. The per-seed table shows the
  consequence directly: on eight of sixteen seeds the arm reproduces the committed grounded
  Hebbian value **exactly** (7.6, 7.6, 13.2, 1.6, 5.6, 1.2, 2.0, 0.4). Share scales with `γ`, so
  the declared grid's top value of 1.0 would have reached about 2.6%; **`γ` would need to be
  roughly 40 for the term to be comparable in magnitude to the Hebbian term**, and the registered
  grid never approached that. D2's failure is therefore weakly informative: it is a fact about the
  grid, not about Oja decorrelation.

## Per-seed, wild-type

| seed | grounded Hebbian | anti-Hebbian | Oja |
|---|---|---|---|
| 1 | 10.0 | 15.6 | 12.4 |
| 2 | 32.8 | 42.0 | 32.0 |
| 3 | 7.6 | 5.2 | 7.6 |
| 4 | 7.6 | 7.2 | 7.6 |
| 5 | 13.2 | 11.2 | 13.2 |
| 6 | 28.4 | **41.6** | 29.6 |
| 7 | 1.6 | 2.8 | 1.6 |
| 8 | 3.2 | 7.2 | 3.2 |
| 9 | **50.4** | **0.0** | **3.6** |
| 10 | 5.6 | 4.0 | 5.6 |
| 11 | 1.2 | 2.0 | 1.2 |
| 12 | 1.6 | 3.2 | 1.6 |
| 13 | 1.2 | 0.8 | 1.2 |
| 14 | 2.0 | 1.2 | 2.0 |
| 15 | 0.4 | 2.4 | 0.4 |
| 16 | 56.8 | 62.8 | 59.2 |
| median | 6.6 | 4.6 | 4.6 |

**One seed decides D1's sign, and it is the comparator's best.** Seed 9 falls from 50.4 to 0.0
under the anti-Hebbian variant and to 3.6 under Oja. Excluding it, the anti-Hebbian arm's mean
delta over the remaining fifteen seeds is **+2.4** rather than −0.9, and it is up on nine of them,
three substantially (seed 6 +13.2, seed 2 +9.2, seed 1 +5.6). That exclusion is **post-hoc and
licenses nothing** — it is recorded because the same bimodal shape has now decided four panels,
and because a registered statistic matched to it is the standing lesson from Logbook 042.

## Reading

- **The prediction fails as registered.** Neither term recovers the loss grounding caused, and
  `full_recovery` is false everywhere. The inhibitory-brake explanation of the sign-grounding
  collapse is not supported by the arm built to test it.
- **The explanation is not refuted so much as unbuildable at this fidelity.** A transmitter-only
  atlas grounds 5.8% of synapses as inhibitory. The animal's inhibition is not only presynaptic
  identity: cholinergic and glutamatergic synapses can be inhibitory through their post-synaptic
  receptor, which is exactly what the receptor layer would add. This result raises the value of
  that layer from fidelity work to a prerequisite for testing this hypothesis properly.
- **The wiring contrast is again unconfirmed** (+2.6 and +2.0, both intervals spanning zero),
  as in every panel since the first.
- **A negative with a measured cause is worth more than a negative without one.** The share
  annotation converted "two terms failed" into "one term did all it could and the other was never
  turned up", and only the first of those is evidence about the hypothesis.

## Campaign facts

- Pilot: 6 runs (3 coefficients × seeds 1–2 × 1000 episodes), 8 minutes, all exit 0. Grid,
  criterion and pin in `launch.md`, written before the test.
- Test: 64 runs (4 arms × seeds 1–16 × 1000 episodes), 42 minutes, all exit 0, no tracebacks.
- Extensions: none needed; every run converged at the budget.
- The anti-Hebbian variant has no hyperparameter, so it was neither piloted nor tuned.
