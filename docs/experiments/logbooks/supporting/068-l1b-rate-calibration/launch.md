# L.1b — L.1's interaction re-read at the rate-matched 0.0001: the registered protocol

Registered in `openspec/changes/add-l1b-rate-calibration`, reviewed and committed **before** any arm
runs. The identity check and the pilot below run first and spend no registered seed.

## The question

[L.1](../../066-l4-readout-width.md) read `pooling_hid_structure` at `plasticity_rate` **0.001**:
interaction **+0.2818** on `auc_success`, the wild type ahead at the per-neuron width (+0.1852) where
at the pooled width the null led (−0.0966). [L.4/L.5](../../067-l4-feature-ablations.md)'s
rate-matched baseline then measured the two wide learning arms at **0.0001** on the same 96 seeds:
wiring effect **−0.0977, null ahead**. L.1's sign-flip form does not hold one decade below its rate.

The rate was inherited: R.2 pinned it on the 8-parameter readout and waived its check;
[L.0](../../064-l4-frozen-features.md) checked it at the pooled width on the wild type alone and
found 0.001 the optimum there (seeds 101–104); L.1 carried it to a 78-parameter readout and to the
null without a sweep.

*(Corrected 2026-09-19, before the campaign, while reading the pilot.)* **L.0's rate-check table
reports the whole-run mean foods; every contrast in this programme reads the plateau-tail mean.**
This record first cited L.0's figures — 17.41 at 0.001 against 12.08 at 0.0001 and 12.81 at 0.01 —
without naming the statistic, which invites reading them against a harness number they are not
commensurable with. Both, on the same runs:

| rate | whole-run mean (L.0's table) | plateau-tail mean (the harness metric) |
|---|---|---|
| 0.0001 | 12.086 | **14.488** |
| **0.001 (committed)** | **17.417** | **18.085** |
| 0.01 | 12.810 | 12.842 |

L.0's ordering and its "tight across seeds" reading hold on either statistic — 0.001 leads at the
pooled width and its spread is 1.05 foods against 6.47 at 0.0001 and 10.52 at 0.01 — so nothing in
L.0 changes. What changes is the margin this record may cite: **0.001 leads 0.0001 by 3.60 foods on
the harness metric, not 5.33**. The 0.001 figures come from `campaigns/readout-width-pilot`, L.1's
pilot of the same config and seeds, which reproduces L.0's published whole-run 17.41 exactly.

> *At 0.0001, does widening the readout still help the wild type more than the shuffle — or was
> L.1's interaction a property of 0.001?*

## Two arms complete a 2×2 that is three-quarters measured

| | pooled (8 params) | per-neuron (78) |
|---|---|---|
| wild type @ 0.0001 | **this campaign** | 0.7013 — `campaigns/feature-ablations` (L.4) |
| rewired null @ 0.0001 | **this campaign** | 0.7990 — `campaigns/feature-ablations` (L.4) |
| floors | `campaigns/readout-width` (L.1); the rate is inert under `freeze_updates` | |

```text
I_1e-4 = (wt_wide − wt_pooled) − (rn_wide − rn_pooled)      per seed at 0.0001, then the paired test
```

Two configs, each one key (`plasticity_rate: 0.0001`) from its committed pooled parent, and one key
(`readout_width`) from the wide arm at the same rate — both asserted by exact-key test:
`..._readout_only_r1e4.yml` (L.0's rate-check config, promoted, header rewritten) and
`..._readout_only_r1e4_rewired_null.yml` (new). **192 runs**, seeds 1–96, 3000 episodes.

**The sign-flip form of L.1 cannot reproduce here, and this is stated before the runs.** The wide
wiring effect at 0.0001 is −0.0977 (null ahead). L.1's *registered* primary was the interaction; the
sign flip was its size at 0.001. A positive here means the pool hides wiring-specific structure at
both rates while the wild type leads at neither width at 0.0001 — the weaker claim, reported as such.

## The minimum effect is a decision rule

`pool_effect_survives_the_rate` requires the interaction at 0.0001 to be significantly positive
**and** `abs(Δ) ≥ 0.141` — **half** of L.1's +0.2818. Below that, a significant interaction is named
*shrunk below half* and reads rate-specific: a claim that has lost more than half its size to a
one-decade rate change is a claim about the rate.

**Sensitivity, from the realised spreads.** The wide wiring effect at 0.0001 has sd 0.135 over 96
seeds (067); the pooled cells at 0.001 spread 0.218–0.228 each; L.1 measured ρ ≈ 0.08 between widths,
so the halves are treated as independent.

| pooled-cell sd assumed | interaction sd | se at n = 96 | detectable at 80% | power at 0.141 |
|---|---|---|---|---|
| 0.22 (as at 0.001) | 0.339 | 0.0346 | 0.097 | **0.98** |
| 0.30 (pessimistic) | 0.445 | 0.0454 | 0.127 | **0.87** |

## The registered readings

| reading | when |
|---|---|
| `pool_effect_survives_the_rate` | interaction significantly **positive** and abs(Δ) ≥ 0.141 |
| `pool_effect_is_rate_specific` | no significant interaction — **a failure to detect**, size and CI carried — **or** significant positive below 0.141, *shrunk below half* |
| `width_favours_the_shuffle_at_this_rate` | interaction significantly **negative**. Reported, not explained |
| `no_learning`, `insufficient_seeds` | a gate fails on a 0.0001 arm; fewer than 5 pairs |

**Eight tests in one BH-FDR family**: the interaction at 0.0001, both main effects there, the
**three-way** `I_1e-3 − I_1e-4` per seed (L.1's committed learning arms re-scored through the same
efficiency call), and four learning gates — each 0.0001 arm against its own-width floor. The priors
are L.1's committed tests on the same floors (q = 0.462 at both widths) and are not re-run. The wide
wiring effect at 0.0001 is a reference outside the family, as 067 committed it.

## Reused cells, and the identity check that licenses them

Three campaigns supply cells. The wide 0.0001 arms ran under the current output controls on
2026-09-18 and nothing in `packages/`, the runner or the configs has changed since; L.1's arms ran
before the controls, and L.4's identity check covered only `wt_wide` and `rn_wide` at 0.001. So
**eight runs at seed 1** — the four floors, L.1's two pooled learning arms, and the two wide 0.0001
arms — are re-run under the current path and compared to their committed logs on every field
`read_log` parses. Identical: reused, and this record says so. Any field differing: that arm's cell
is re-run for seeds 1–96 and nothing of it is reused. L.1's wide learning arms at 0.001 were checked
by L.4 and are not re-checked.

**Disk.** A run under the controls writes ~17.1 MB of exports plus a ~0.5 MB campaign log (067's
measurement); 192 runs need **~3.4 GB**, the 16 check/pilot runs ~0.3 GB.

## The stop clauses

1. **Pilot on seeds 101–104**, eight runs, under the controls: both pooled 0.0001 arms learn above
   L.1's pilot pooled floors (`campaigns/readout-width-pilot`), and each differs from the 0.001
   pooled pilot arm of the same wiring. The wild-type arm is compared to L.0's rate-check run of the
   same config on the same seeds (`campaigns/frozen-features-rate`; 14.488 plateau-tail mean foods,
   the figure the harness reads), which is one more identity point across the flag change and the
   header rewrite. No reading at four pairs.
2. The identity check passes, or the affected cell is re-run.
3. This record committed before anything runs.

## The pilot, 2026-09-19 — pass

8/8 runs succeeded in 1721 s. Plateau-tail mean foods, against L.1's pilot floors and its 0.001
pooled arms at the same seeds:

| wiring | seed | @0.0001 | its floor | gain | @0.001 |
|---|---|---|---|---|---|
| wt | 101 | 15.217 | 1.089 | **+14.128** | 18.697 |
| wt | 102 | 10.107 | 2.571 | **+7.536** | 17.647 |
| wt | 103 | 16.580 | 2.072 | **+14.508** | 18.075 |
| wt | 104 | 16.049 | 1.584 | **+14.465** | 17.921 |
| rn | 101 | 17.169 | 6.149 | **+11.020** | 19.505 |
| rn | 102 | 14.072 | 1.864 | **+12.208** | 15.767 |
| rn | 103 | 17.648 | 2.609 | **+15.039** | 19.425 |
| rn | 104 | 18.321 | 5.852 | **+12.469** | 19.484 |

All three registered conditions hold. **Both arms learn**: mean gain over floor **+12.659** (wild
type) and **+12.684** (null), no seed below its floor. **Both differ from their 0.001 partners**, on
every one of the four seeds and in the direction L.0's pooled-width check predicted — the lower rate
is *worse* at the pooled width, by 2.3 (wild type) and 2.1 (null) foods. **The wild-type arm
reproduces L.0's rate-check runs exactly**: identical on all eight fields `read_log` derives from the
log and the experiment record, across both the output-control flags and the config header rewrite.

**One field could not be compared, and it is named rather than counted as a pass.**
`peak_action_density` reads `session/data/tracking_actions.csv` from a run's export, and **L.0's
2026-09-15 exports have since been deleted**, so it is `None` on the committed side and a number on
the new side for all four seeds. The difference is a property of the deleted export, not of the runs,
and the field enters no contrast in this harness. This does **not** touch the registered identity
check of clause 2: all eight of its committed runs still have their exports and their
`tracking_actions.csv`, so that check compares all nine fields with nothing missing.

**Not read at four seeds, and recorded so it cannot be read later as if it had been**: at the pooled
width the null's gain over floor matched the wild type's (+12.68 against +12.66), where at 0.001 L.1
found the pooled null *ahead* on `auc_success`. Whether that holds at 96 seeds is what the campaign
measures.

## The identity check, 2026-09-19 — pass

8/8 runs succeeded in 2227 s. Each reused arm's seed-1 run, re-run under the current path
(`--no-detailed-export --no-file-log` on today's tree), against its committed log on **every one of
the nine fields `read_log` parses** — `success`, `foods`, `episodes`, `converged`, `onset`,
`evasion_rate`, `temp_comfort`, `curve`, `peak_action_density`:

| reused arm | committed in | plateau-tail foods | result |
|---|---|---|---|
| `wt_pooled_frozen` | `campaigns/readout-width` (L.1) | 11.480 | identical, 9/9 fields |
| `rn_pooled_frozen` | `campaigns/readout-width` | 6.427 | identical, 9/9 |
| `wt_wide_frozen` | `campaigns/readout-width` | 11.480 | identical, 9/9 |
| `rn_wide_frozen` | `campaigns/readout-width` | 6.417 | identical, 9/9 |
| `wt_pooled` @ 0.001 | `campaigns/readout-width` | 19.280 | identical, 9/9 |
| `rn_pooled` @ 0.001 | `campaigns/readout-width` | 19.604 | identical, 9/9 |
| `wt_wide` @ 0.0001 | `campaigns/feature-ablations` (L.4) | 19.607 | identical, 9/9 |
| `rn_wide` @ 0.0001 | `campaigns/feature-ablations` (L.4) | 19.421 | identical, 9/9 |

**No field differs on any arm, so every reused cell is licensed and nothing is re-run.** The
`peak_action_density` gap the pilot met does not arise here: all eight committed runs still hold their
exports and their `session/data/tracking_actions.csv`, checked before the comparison rather than
discovered in it. The two floors at seed 1 agreeing to three decimals across widths is L.1's committed
property — the widths are the same policy at initialisation and differ by ~1e-8 — and is not new
evidence.

Clause 2 is settled: L.1's four floors and two pooled learning arms, and L.4's two wide 0.0001 arms,
are the panel's reused cells on this evidence.

## What the reading conditions, and what it does not decide

L.1's verdict stands as read at 0.001. Each reading adds a dated condition beside it in the tracker,
Logbook 066 and the roadmap: *the pool hid structure at both rates; the wild type's lead at the
per-neuron width is a 0.001 result* — or — *the pool hid structure at 0.001 only*. Neither decides
the rate: both wirings learn better at 0.0001, so 0.001 is the worse operating point for either, and
which rate a rung after 7 runs at is that rung's calibration to make. Nothing is ablated against the
0.0001 baseline here; L.4 did that inside the campaign that established the baseline, and the
[protocol](../../../../research/phase-protocol.md) now forbids it.

## The honest prior

**`pool_effect_is_rate_specific`.** For the interaction to hold at 0.141, the pooled null would have
to lead the pooled wild type by at least **0.24** at 0.0001, against 0.097 at 0.001. The null gained
+0.44 from the rate at the wide width and may gain at the pooled width too; but the pooled wild type
is known to lose ~5 foods at 0.0001 (L.0's check), and that pushes the pooled wiring effect the same
way as the wide one moved — toward the null, but from a smaller lead. Registered before any arm runs.

## What no reading would license

Not a mechanism; not an endpoint claim; not a read-across to block V; not a choice of rate; not a
rewrite of L.1's verdict; and not evidence about the pooled-width null at 0.001, which L.0 and L.1
measured and this does not re-run.

## Reproduce

```bash
P=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop
FLAGS="--theme headless --track-experiment --no-detailed-export --no-file-log"

# 1. the identity check that licenses reusing L.1's floors and pooled arms and L.4's wide 0.0001 arms
uv run python scripts/run_campaign.py \
  --config ${P}_frozen.yml --config ${P}_frozen_rewired_null.yml \
  --config ${P}_frozen_wide.yml --config ${P}_frozen_wide_rewired_null.yml \
  --config ${P}_readout_only.yml --config ${P}_readout_only_rewired_null.yml \
  --config ${P}_readout_only_wide_r1e4.yml --config ${P}_readout_only_wide_r1e4_rewired_null.yml \
  --seeds 1 --runs 3000 --output-dir campaigns/rate-calibration-identity -- $FLAGS

# 2. pilot on DISJOINT seeds 101-104 -- 8 runs, no reading at four pairs
uv run python scripts/run_campaign.py \
  --config ${P}_readout_only_r1e4.yml --config ${P}_readout_only_r1e4_rewired_null.yml \
  --seeds 101-104 --runs 3000 --output-dir campaigns/rate-calibration-pilot -- $FLAGS

# 3. the registered arms -- 192 runs on seeds 1-96
uv run python scripts/run_campaign.py \
  --config ${P}_readout_only_r1e4.yml --config ${P}_readout_only_r1e4_rewired_null.yml \
  --seeds 1-96 --runs 3000 --output-dir campaigns/rate-calibration -- $FLAGS

# 4. score: the 2x2 at 0.0001 beside L.1's at 0.001
uv run python scripts/analysis/l4_rate_calibration.py --campaign campaigns/rate-calibration \
  --wide-rate campaigns/feature-ablations --baseline campaigns/readout-width \
  --out docs/experiments/logbooks/supporting/068-l1b-rate-calibration/rate_calibration.json \
  --csv docs/experiments/logbooks/supporting/068-l1b-rate-calibration/per-seed.csv
```
