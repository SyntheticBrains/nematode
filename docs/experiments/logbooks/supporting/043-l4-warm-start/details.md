# Warm-start panel details

Analysis by `scripts/analysis/l4_warm_start.py` from the manifest beside this file (96 logs; the 25
extended runs replace their shorter logs; attempt 1 of the plastic-set learning arms is set aside,
see `attempt1-plastic.md`), with the random-initialisation floor from
`supporting/041-l4-panel2/per-seed.csv`, the teacher's ceiling from `teacher.json` and the clone
fits from `clones.json`. `panel.json` is the full output; `per-seed.csv` and `curves.csv` are the
per-run table and learning curves.

## Verdict: `sanity_floor_fail`, with `rule_destroys_clone`

| test | contrast | mean Δ | 80% CI | q | +seeds | result |
|---|---|---|---|---|---|---|
| W1 | wt_clone_frozen − random wt_frozen | +31.0 | +23.4 … +38.9 | .023 | 8/8 | **pass** (the gate) |
| W2 | wt_clone_frozen − rn_clone_frozen | −4.0 | −7.9 … −0.4 | .996 | 2/8 | reverse |
| W3 | wt_clone_plastic − rn_clone_plastic | −7.2 | −15.5 … +1.2 | .996 | 3/8 | fail (primary) |
| W4 | wt_clone_plastic − wt_clone_frozen | −25.7 | −34.7 … −17.1 | .996 | 2/8 | reverse |
| W5 | wt_clone_plastic − wt_clone_hebbian | −25.7 | −34.1 … −17.7 | .996 | 1/8 | reverse |
| W6 | wt_fullclone_ppo − wt_ppo | −34.1 | −45.4 … −23.1 | .996 | 1/8 | reverse |

Annotations: `wild_type_holds_better` false, `warm_start_helps_ppo` false, `rule_destroys_clone`
**true**. Every family test is complete (eight seeds).

## Per-arm plateau tails (%), seeds 1–8, sorted

| arm | mean | sorted per seed | competent | of ceiling (98.7) |
|---|---|---|---|---|
| wt_clone_frozen | 38.7 | 21 23 33 39 40 44 47 61 | 8/8 | 0.39 |
| rn_clone_frozen | 42.8 | 26 36 41 44 45 47 48 55 | 8/8 | 0.43 |
| wt_clone_hebbian | 38.6 | 3 7 22 39 50 56 58 74 | 6/8 | 0.39 |
| rn_clone_hebbian | 17.9 | 0 1 2 14 24 25 29 47 | 4/8 | 0.18 |
| wt_clone_plastic | 13.0 | 0 4 5 7 8 11 27 42 | 2/8 | 0.13 |
| rn_clone_plastic | 20.1 | 1 3 4 5 9 30 51 57 | 3/8 | 0.20 |
| wt_fullclone_frozen | 73.7 | 65 65 69 71 77 77 82 83 | 8/8 | 0.75 |
| rn_fullclone_frozen | 74.7 | 56 66 72 77 80 80 81 85 | 8/8 | 0.76 |
| wt_fullclone_ppo | 34.5 | 0 8 35 35 38 52 53 56 | 6/8 | 0.35 |
| rn_fullclone_ppo | 25.4 | 5 7 13 14 31 33 47 51 | 4/8 | 0.26 |
| wt_ppo (low noise, scratch) | 68.5 | 30 54 56 79 80 81 83 85 | 8/8 | 0.69 |
| rn_ppo (low noise, scratch) | 81.2 | 56 78 81 83 85 87 90 91 | 8/8 | 0.82 |

Random-initialisation wild-type frozen floor on the same seeds (panel 2): 36.0, 0.7, 6.7, 12.7,
1.3, 0.7, 2.7, 1.3 (mean 7.8).

## Named descriptive pairs (uncorrected)

| pair | mean Δ | 80% CI | +seeds |
|---|---|---|---|
| wt_fullclone_frozen − rn_fullclone_frozen | −1.0 | −7.2 … +4.7 | 3/8 |
| wt_fullclone_frozen − wt_clone_frozen | +34.9 | +27.6 … +41.9 | 8/8 |
| wt_fullclone_ppo − rn_fullclone_ppo | +9.1 | −6.1 … +24.6 | 6/8 |
| wt_ppo − rn_ppo | −12.6 | −17.9 … −7.5 | 0/8 |
| rn_clone_plastic − rn_clone_frozen | −22.6 | −30.0 … −14.7 | 1/8 |
| rn_clone_plastic − rn_clone_hebbian | +2.2 | −10.6 … +15.1 | 5/8 |
| rn_fullclone_ppo − rn_ppo | −55.8 | −66.4 … −45.1 | 0/8 |
| wt_clone_hebbian − rn_clone_hebbian | +20.7 | (every remaining pair is in `panel.json`) | — |

## Learning curves, block means over the eight seeds (first 2000 or 3000 episodes)

| arm | 250 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|---|
| wt_clone_hebbian | 35 | 35 | 39 | 39 | 38 | | |
| rn_clone_hebbian | 14 | 15 | 17 | 17 | 18 | | |
| wt_clone_plastic | 14 | 24 | 23 | 22 | 13 | | |
| rn_clone_plastic | 11 | 14 | 20 | 25 | 15 | | |
| wt_fullclone_ppo | 65 | 52 | 39 | 44 | 31 | 38 | 29 |
| rn_fullclone_ppo | 62 | 37 | 31 | 27 | 27 | 25 | 24 |
| wt_ppo | 42 | 53 | 65 | 64 | 68 | 69 | 69 |
| rn_ppo | 29 | 55 | 61 | 60 | 67 | 71 | 72 |

(Blocks beyond an arm's budget average only the seeds that were extended and are omitted.)

## Clone fits

Held-out loss means: plastic-set wild-type 0.310 (6 of 8 flagged weak), plastic-set rewired 0.303
(5 flagged), full-set wild-type 0.262 (0 flagged), full-set rewired 0.262 (1 flagged). The
constant-predictor baseline is 0.454 and the teacher's own sampling noise 0.311
(`clone-fit-notes.md`, written before the panel ran).

## Reading

- **The clone gate passes on every seed.** Cloning the chemical weights alone, behind the fixed
  anatomical readout, turns an 8% random policy into a 39% one on every seed; cloning every
  parameter PPO trains reaches 74%, three quarters of the teacher's 99%. The connectome can hold a
  competent policy, and the earlier panels' floors were about initialisation, not capacity.
- **The wiring does not limit representability.** The rewired null holds the policy at least as
  well as the wild-type under both parameter sets (W2 reverses; the full-set pair is a tie).
- **The three-factor rule takes a competent policy apart, on both wirings.** From a 39% clone the
  wild-type arm ends at 13% and from 43% the rewired arm at 20%; W4 reverses by 26 points with the
  interval far from zero. The curves show a brief rise in the first thousand episodes before the
  collapse. This is the rule as loaded, after the load defect of attempt 1 was fixed and verified.
- **The unmodulated Hebbian rule holds the clone on the wild-type and not on the rewired null.**
  The wild-type Hebbian arm ends at 39%, level with its frozen clone, with six of eight seeds
  competent and three above 55%; the rewired arm falls to 18%. The 21-point wiring difference is
  the one wiring-specific effect in the panel, and it is descriptive: it sits in the Hebbian
  floor, as in panels 1–3, not in the reward-modulated arm.
- **A warm start hurts PPO.** Fine-tuning from the 74% full-set clone declines from the first
  block (65 → 29 on the wild-type, 62 → 24 on the rewired null) while PPO from random weights at
  the same initial noise climbs to 69% and 81%. W6 reverses by 34 points. The PPO update from a
  fresh critic pulls the cloned policy down faster than it rebuilds it.
- **Low initial noise alone lifts connectome PPO.** From random weights at `initial_log_std −1.0`
  the wild-type reaches 69% and the rewired null 81% in 3000 episodes, against 52% at 6000
  episodes with std 1.0 noise in Logbook 029. The rewired null beats the wild-type on all eight
  seeds (−12.6, interval clear of zero), descriptively: under gradient learning the specific
  wiring is, again, not an advantage.
- **Extensions.** Twenty-five runs received the single registered extension (four frozen at 900,
  fourteen PPO at 4500, seven plastic-set at 3000); eight remained non-converged after it and are
  read on their extended plateau tails as the registration prescribes. The family and the
  verdict were the same before and after the extensions.

## Campaign facts

- Teacher: 8 runs at 6000, 13:31–14:03; seed 7 selected (95.5%); recording 300 episodes at seed
  101, ceiling 98.7%; 32 clones.
- Panel: frozen 14:53–15:07; plastic-set attempt 1 15:07–16:07 (superseded); PPO 16:07–16:57;
  plastic-set re-run 16:57–18:10; extensions 18:10–20:05. All on 16 workers; every runner exited
  0; no tracebacks.
