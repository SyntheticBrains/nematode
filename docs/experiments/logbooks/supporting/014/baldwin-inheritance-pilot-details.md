# 014 supporting: Baldwin Inheritance Pilot — Details

Detail-level data and discussion for the M4 logbook. The main logbook
(`docs/experiments/logbooks/014-baldwin-inheritance-pilot.md`) contains
the headline findings and the **STOP** verdict; this file contains
per-seed tables, the evolved-hyperparameter distributions, the F1
innate-only forensic discussion, and the wall-time breakdown.

## Per-seed final-fitness tables

### Baldwin pilot

| Seed | Final gen (early-stopped at) | Best fitness | Population mean (final gen) |
|---|---|---|---|
| 42 | 7 | 0.88 | 0.43 |
| 43 | 15 | 0.96 | 0.44 |
| 44 | 11 | 0.92 | 0.53 |
| 45 | 7 | 0.92 | 0.22 |

### Lamarckian rerun

| Seed | Final gen | Best fitness |
|---|---|---|
| 42 | 7 | 1.00 |
| 43 | 11 | 1.00 |
| 44 | 10 | 1.00 |
| 45 | 11 | 1.00 |

All 4 lamarckian seeds reach 1.00 — reproduces M3 exactly.

### Control (M3-control rerun on M4 revision)

| Seed | Final gen | Best fitness |
|---|---|---|
| 42 | 6 | 0.84 |
| 43 | 9 | 0.96 |
| 44 | 9 | 0.96 |
| 45 | 12 | 0.96 |

Seed 42 saturated at 0.84 — never reached the Lamarckian-style 0.88 plateau the M3 control did. Probably noise within the seed's fitness landscape; doesn't change the across-seed mean materially.

## Per-seed gen-to-0.92 trajectories

Source: `artifacts/logbooks/014/m4_baldwin_pilot/summary/convergence_speed.csv`.

```csv
seed,baldwin_gen_to_092,lamarckian_gen_to_092,control_gen_to_092,f1_innate_only_success_rate
42,,3,,0.000000
43,8,4,5,0.000000
44,7,4,5,0.000000
45,3,7,8,0.000000
```

Aggregator's gen-to-092 means use a `max_gens + 1` fallback for seeds that never reach the threshold:

- Baldwin: [16 (fallback for seed 42), 8, 7, 3] → mean 8.50
- Lamarckian: [3, 4, 4, 7] → mean 4.50
- Control: [16 (fallback for seed 42), 5, 5, 8] → mean 8.50

## Evolved-hyperparameter distributions

Decoded `best_params.json` per seed, against the Baldwin pilot's 6-field schema:

| Seed | actor_lr | critic_lr | gamma | entropy_coef | weight_init_scale | entropy_decay_episodes | best_fitness |
|---|---|---|---|---|---|---|---|
| 42 | 7.34e-04 | 1.13e-04 | 0.934 | 1.04e-04 | 1.22 | 1465 | 0.88 |
| 43 | 6.75e-04 | 5.20e-05 | 0.934 | 2.51e-03 | 1.33 | 1284 | 0.96 |
| 44 | 7.61e-04 | 8.30e-05 | 0.991 | 9.04e-04 | 1.07 | 1022 | 0.92 |
| 45 | 9.67e-04 | 4.26e-04 | 0.934 | 7.69e-04 | 0.57 | 1562 | 0.92 |
| (default) | n/a | n/a | n/a | n/a | 1.00 | 500 | n/a |

Notable for the M4 hypotheses:

- **`weight_init_scale`**: TPE evolved to non-default values across all 4 seeds (range 0.57-1.33). Risk 1 of the design doc canaried whether TPE would stay near 1.0; data falsifies that risk. The field IS being explored; it just doesn't help. Seed 45's 0.57 sits at the schema's lower edge [0.5, 2.0] — TPE actively searching toward smaller-init.
- **`entropy_decay_episodes`**: TPE consistently evolved upward (1022-1562 across all 4 seeds vs the 500 brain default). Suggests TPE wants slower entropy decay than the M3 hand-set value. Doesn't help convergence speed in absolute terms vs the M3 control's 4-field schema (which holds `entropy_decay_episodes` at the brain default 500), but the directional signal is consistent.
- **Carried-over fields** (actor_lr, critic_lr, gamma, entropy_coef): broadly similar to M3's lamarckian-arm + control-arm evolved values. No surprise — these are the dominant fields in the K-train phase.

## F1 innate-only forensic discussion

All 4 seeds produced exactly 0 successful frozen-eval episodes (out of 25 episodes per seed). The hand-tuned baseline produces ~17% — meaning the Baldwin elite genome alone is *worse* than hand-tuning, not better.

Plausible mechanism: TPE-evolved `actor_lr` (~7e-4 across seeds), `entropy_coef` (1e-4 to 2.5e-3), and `weight_init_scale` (0.57-1.33) are tuned to interact with the K=50 train phase's gradient updates. Without those updates, the random-init policy under those hyperparams produces an action distribution biased toward exploitation of a fresh-init (nearly-uniform) value head — but the value head hasn't seen any environment data, so the "exploitation" is essentially random. Combined with low entropy, the policy gets stuck in non-productive action sequences quickly.

In contrast, the hand-tuned baseline uses `actor_lr=3e-4`, `entropy_coef=2e-2`, `weight_init_scale=1.0` (default), and `entropy_decay_episodes=800` — values that produce a slightly higher initial action entropy, which under random-init weights is more like uniform random than overconfident. Random uniform exploration is, on this arm, modestly better than overconfident wrong exploitation.

The 0.0 vs 0.17 gap means the genetic-assimilation gate fails decisively (-0.17 vs need ≥+0.10). The original threshold (+0.10pp over baseline) was calibrated for what the design doc called "noticeably better than hand-tuning". The actual data says "noticeably worse than hand-tuning". The gate calibration was correct directionally but the wrong direction won.

If a future Baldwin variant wants to exhibit genetic assimilation, it would need to either (a) evolve hyperparams that produce reasonable behaviour even WITHOUT training (e.g. larger entropy_coef forcing more exploration regardless of value-head quality), or (b) actually transfer some learned state — at which point it's no longer pure-trait Baldwin.

## Wall-time breakdown

Pilot start: 09:57. All 3 arms launched in parallel (3 scripts in background, each using 4 internal workers; 12 worker procs against 10 cores).

| Arm | Start | End | Wall |
|---|---|---|---|
| Control rerun | 09:58 | ~10:55 | ~57 min |
| Baldwin pilot | 09:57 | ~11:00 | ~63 min |
| Lamarckian rerun | 09:58 | ~11:05 | ~67 min |
| F1 post-pilot | 11:05 | ~11:08 | ~3 min |
| Total wall (pilot + F1) | 09:57 | 11:08 | **~71 min** |

Per-seed wall-time variance under parallel contention was significant:

- Lamarckian seed-43 took longest within its arm (~17 min — saturated late at gen 11).
- Baldwin seed-43 also long (~16 min — gen 15 before early-stop fired).
- Saturated-fast seeds (Baldwin 42, 45 at gen 7) finished in ~10-11 min each.

`early_stop_on_saturation: 5` saved roughly half the per-seed wall on saturating arms; without it, every seed would have run the full 20 generations.

## Schema confounder check

The Baldwin pilot evolves 6 fields; the M3 control's 4. Did the speed-gate FAIL because the new evolvable knobs didn't help, or because the control's 4-field schema was "missing" critical degrees of freedom that Baldwin's extra fields couldn't compensate for?

Cross-check on the 4 retained fields (actor_lr, critic_lr, gamma, entropy_coef): Baldwin's evolved values for those 4 fields are broadly comparable to M3's published lamarckian-arm + control-arm evolved values. The two new fields (`weight_init_scale`, `entropy_decay_episodes`) added exploration room but didn't change the speed metric.

So the answer is "the new knobs didn't help" — not "the schema was inappropriate". The Baldwin signal is genuinely absent on this arm, given this fitness landscape and this K=50 budget.

## Cross-arm code-revision check

The Lamarckian and control reruns are on the M4 code revision; M3's published numbers were on the M3 revision.

| Arm | M3 published | M4 rerun | Delta |
|---|---|---|---|
| Lamarckian mean gen-to-0.92 | 4.50 | 4.50 | 0.00 |
| Lamarckian per-seed | [3, 4, 4, 7] | [3, 4, 4, 7] | identical |
| Lamarckian best at gen 20 (per seed) | all 1.00 | all 1.00 | identical |
| Control mean gen-to-0.92 | 9.75 | 8.50 | -1.25 (faster) |

The Lamarckian rerun reproduces M3 exactly — confirming the M4 code revision (kind() Protocol method + two-guard split + early-stop monitor + weight_init_scale brain field default 1.0) is byte-equivalent for the M3 lamarckian path.

The control rerun is slightly faster than M3's published number (8.50 vs 9.75), plausibly because `--early-stop-on-saturation 5` truncates the late-gen flat plateau that pulled M3's published mean upward (M3 ran the full 20 generations; gens 13-20 of seeds 43/44/45 contributed 0 lift but added to the gen-to-0.92 fallback when the never-reached fallback was `max_gens + 1 = 21`). This is a slight M4-side improvement, not a regression.

If the M4 control had been *slower* than M3, that would be a confounder — Baldwin's failure to beat the M4 control might have hidden a real Baldwin-vs-M3-control speed lift. Instead the M4 control is faster, strengthening the "no Baldwin signal" finding.
