# 043: The Warm-Start Panel — What a Local Rule Does From a Cloned Competent Policy (7a-i / S.2 / Phase 7)

**Status**: completed — **`sanity_floor_fail`** with **`rule_destroys_clone`** under the
pre-registered verdict map. Three panels had shown that on the C3 cell the random initial
weights decide what any local rule can do; this one removed that confound by cloning one
competent policy into the connectome on every seed and asking what happens next. The clone
gate passes on every seed: behavioural cloning into the chemical weights alone, behind the fixed
anatomical readout, turns an 8% random policy into a **39%** one, and cloning every parameter PPO
trains reaches **74%**, three quarters of the teacher's 99%, and the degree-preserving rewired
null carries it at least as well. The connectome can therefore *hold* a cloned copy of this one
teacher's policy at that level, so the fifth-of-six rank of
[Logbook 029](029-continuous-architecture-ranking.md) is not explained by an inability to carry a
policy of that quality on either wiring — a claim about one cloned policy, not about the policies
the substrate was never given or about its access to them under learning. From that start the
reward-modulated three-factor rule falls **below its own frozen clone** on the wild-type
(W4 **−25.7**, registered) and on the rewired null (**−22.6**, CI[−30.0, −14.7], 1/8,
descriptive), and below its Hebbian clone on the wild-type (W5 **−25.7**, registered) though not
on the rewired null (+2.2, not significant). The unmodulated Hebbian rule holds the clone on the
wild-type (39%, six of eight seeds competent) and not on the rewired null (18%), and PPO
fine-tuning from the 74% clone loses by **34 points** to PPO from random
weights at the same initial noise, which itself reaches 69% and 81% in 3000 episodes against
029's 52% in 6000. One defect surfaced mid-panel and was fixed under a dated amendment.

**Branch**: `feat/l4-warm-start-tooling` (PR #324, #325), `feat/l4-warm-start-panel` (PR #326).

**Date**: 2026-09-08.

**OpenSpec changes**: `add-l4-warm-start-tooling` (connectome weight persistence, the reported
action mean, rollout recording, the cloning trainer; new capability `behavioural-cloning`) and
`add-l4-warm-start-panel` (the registration; extends `l4-plasticity-panel`, adds the `{seed}`
placeholder to `weight-persistence`); both archived.

## Objective

Roadmap D13's imitation warm start, re-aimed after Logbooks 040–042: (1) can the connectome
*hold* a competent policy at all, and does the specific wiring matter for holding it; (2) what
does the reward-modulated rule do from a competent start, against the same floors panel 1 used;
(3) does a warm start make PPO on the connectome competitive.

## Background

Panels 1–3 ([040](040-l4-panel.md), [041](041-l4-panel2.md), [042](042-l4-panel3.md)) resolved
the same way: outcomes are fixed points seeded by the random initial weights, most random
initialisations are dead, and the reward-modulated rule neither beats its floors nor separates
the wirings at the sample sizes run. Every result had been read against a prior that was mostly
dead. Ratified with Chris 2026-09-07: one teacher (the best of eight MLP-PPO seeds), two clones
per student (the plastic set the rule can inherit, and the full PPO set), the warm-started
plastic 2×2 with its floors, plus PPO fine-tuning against PPO from scratch.

## Hypothesis

Pre-registered before any run (`supporting/043-l4-warm-start/launch.md` committed first): six
one-sided paired tests corrected together — **W1** the wild-type plastic-set frozen clone over
panel 2's random-initialisation frozen floor on the same seeds (the gate); **W2** wild-type over
rewired plastic-set frozen clone (representability); **W3** wild-type over rewired three-factor
from the clone (the primary); **W4** three-factor over its frozen clone; **W5** three-factor over
its Hebbian clone; **W6** PPO from the full-set clone over low-noise PPO from random weights. The
verdict in order: `insufficient_seeds`, `clone_fail`, `sanity_floor_fail`,
`rewired_beats_wild_type`, `specific_wiring`, `degree_statistics`, `inconclusive`; W2 and W6
annotate; `rule_destroys_clone` is named when W4's interval lies entirely below zero. Power at
n = 8 was stated in advance: a test passes only with seven or eight concordant seeds.

## Method

**Tooling** (PR #324, #325): the connectome brain implements weight persistence (a `topology`
component with every parameter and the wiring buffers, never the per-episode trace buffers; a
load checks the std mode and the wiring before touching anything); continuous brains report the
noiseless action mean beside the sample; `--record-rollouts` writes one JSON line per step; and
`scripts/campaigns/l4_behavioural_clone.py` fits a student's squashed action mean to a recorded
teacher's by action-space mean squared error over the `plastic` set (chemical weights, masked to
the wiring) or the `full` set (everything PPO trains except the noise, which a mean-only record
cannot target).

**Teacher**: `mlpppo_small_continuous2d_combined_klinotaxis` on seeds 1–8 at 6000 episodes; seed
7 selected (95.5% plateau tail); recorded frozen for 300 episodes at seed 101 (103,827 steps);
ceiling 98.7%. The teacher's policy is bang-bang: speed 1.0 on every step, |turn| > 0.95 on 87%.

**Clones**: per seed and wiring, `plastic` (student: the plastic frozen arm) and `full` (student:
a one-key low-noise PPO derivation, `initial_log_std −1.0`, so the clone's saved noise matches
the plastic arms'), 300 epochs, learning rate 1e-3, batch 256, holdout 0.2. Every clone plateaus
near a held-out loss of 0.30 (plastic set) or 0.26 (full set) against a constant-predictor
baseline of 0.454 and the teacher's own sampling noise of 0.311; 11 of 16 plastic-set clones and
1 of 16 full-set clones are flagged weak by the registered criterion. The reading was written
before the panel ran (`clone-fit-notes.md`).

**Arms**: twelve on paired seeds 1–8 — from the plastic-set clone the frozen, Hebbian and
three-factor arms on both wirings (600, 2000, 2000 episodes); from the full-set clone the frozen
arm (600) and PPO (3000) on both wirings; low-noise PPO from random weights on both wirings
(3000). One registered extension at 1.5× for a run the plateau detector marks non-converged;
twenty-five were applied, eight stayed non-converged and are read on their extended tail. No
pilot. Harness `scripts/analysis/l4_warm_start.py`.

**The amendment.** The first run of the four plastic-set learning arms found the rule taking the
clone apart on step one: 7.7% in the first block against the frozen clone's 37.7%, episode 1
dead at step 90. The cause was a defect in the load path, not the rule: the rule's homeostatic
norm targets were the incoming norms at construction (random initialisation, median 0.95) and
were not refreshed on load, while the clone's norms sat six times higher (median 5.4), so the
first plastic step rescaled every unit back to its random-init norm. Verified directly, fixed
(the load re-anchors the targets to the loaded weights; the persistence spec and its test say
so), ratified as a dated amendment, the run kept as attempt 1 (`attempt1-plastic.md`), and the
four arms re-run. The frozen arms take no rule step and the PPO arms run no homeostasis, so they
stand.

## Results

### The registered family (paired seeds 1–8, BH-FDR α = 0.05)

| test | contrast | mean Δ | 80% CI | q | +seeds | result |
|---|---|---|---|---|---|---|
| W1 | wt_clone_frozen − random wt_frozen | +31.0 | +23.4 … +38.9 | .023 | 8/8 | **pass** |
| W2 | wt_clone_frozen − rn_clone_frozen | −4.0 | −7.9 … −0.4 | .996 | 2/8 | reverse |
| W3 | wt_clone_plastic − rn_clone_plastic | −7.2 | −15.5 … +1.2 | .996 | 3/8 | fail |
| W4 | wt_clone_plastic − wt_clone_frozen | −25.7 | −34.7 … −17.1 | .996 | 2/8 | reverse |
| W5 | wt_clone_plastic − wt_clone_hebbian | −25.7 | −34.1 … −17.7 | .996 | 1/8 | reverse |
| W6 | wt_fullclone_ppo − wt_ppo | −34.1 | −45.4 … −23.1 | .996 | 1/8 | reverse |

**Verdict: `sanity_floor_fail`**; annotations `rule_destroys_clone` **true**,
`wild_type_holds_better` false, `warm_start_helps_ppo` false.

### Per-arm plateau tails (%), seeds 1–8

| arm | mean | sorted per seed | competent | of ceiling |
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

Descriptive: wt_clone_hebbian − rn_clone_hebbian **+20.7**; wt_ppo − rn_ppo **−12.6**
(CI[−17.9, −7.5], 0/8); wt_fullclone_frozen − rn_fullclone_frozen −1.0; wt_fullclone_frozen −
wt_clone_frozen +34.9 (8/8). Curves and every pair in `supporting/043-l4-warm-start/`.

## Analysis

1. **The connectome can retain a competent policy, on either wiring.** With only the chemical
   weights free behind a readout that is anatomy rather than a trained decoder, the clone scores
   39% on every seed; with gains and readout free it scores 74%. The rewired null carries the
   policy at least as well under both sets. What this settles is retention of the one policy it
   was given: the substrate can express and run a policy of that quality, so the fixed points of
   panels 1–3 are not explained by an inability to hold one. It does not establish that the
   connectome could represent a better policy, nor that its rank under PPO has no
   representability component at all — one cloned teacher is one point in policy space.
2. **The three-factor rule takes a competent policy apart.** From a 39% clone the wild-type arm
   rises briefly in the first thousand episodes and then collapses to 13%; the rewired arm from
   43% to 20%. Only the wild-type contrasts are registered (W4, W5); the rewired equivalent is
   descriptive — below its frozen clone by −22.6 (CI[−30.0, −14.7], 1 of 8 seeds positive) and
   level with its Hebbian clone (+2.2, not significant) — so "on both wirings" holds for the
   frozen comparison and is untested for the reward contrast. This is the rule acting on the
   clone as loaded, after the load defect was fixed and verified. The registered map names it:
   `rule_destroys_clone`. The rule's
   reward-modulated updates do not preserve a policy that reward alone would reward; they move
   the weights toward the rule's own fixed point, which on this cell is worse than the clone.
3. **The Hebbian rule holds the clone on the wild-type and not on the scramble.** The
   unmodulated rule ends level with the frozen clone on the wild-type (39%, six of eight seeds
   competent, three above 55%) and at 18% on the rewired null. A 21-point wiring difference, the
   only wiring-specific effect in the panel, and once again it lives in the Hebbian floor rather
   than in the reward-modulated arm — the same signal panels 1–3 chased and could not confirm,
   here on a competent start rather than a random one, and descriptive.
4. **A warm start hurts PPO, and low noise helps it.** Fine-tuning from the 74% clone declines
   from its first block (65 → 29 on the wild-type, 62 → 24 on the rewired) while PPO from random
   weights at the same initial noise climbs to 69% and 81%. A fresh critic and clipped updates on
   a cloned policy pull it down faster than they rebuild it. Separately, starting connectome PPO
   at `initial_log_std −1.0` instead of 0 lifts it from 029's 52% at 6000 episodes to 69–81% at
   3000: exploration noise at std 1.0 had been capping the PPO arm as it capped the plastic ones.
   Under gradient learning the rewired null beats the wild-type on all eight seeds.
5. **The clone fits plateau where the substrate saturates.** Both parameter sets stop near a
   held-out loss of 0.3 against a 0.45 baseline, inflating weight norms five- to seven-fold to
   reproduce a switch through tanh units. That plateau is what made attempt 1's defect
   decisive, and it is what a sign-aware or sparser substrate would have to move.

## Conclusions

- The connectome retains a cloned competent policy on either wiring, at 39% of the ceiling
  through the chemical weights alone and 74% with every PPO parameter. D13's representability
  question is answered for a policy of that quality; nothing here speaks to policies the
  substrate was never handed.
- The minimal reward-modulated three-factor rule, as registered, destroys a competent policy on
  the wild-type wiring (registered) and on the rewired null (descriptive). From a competent start as from a random one, the rule's fixed points are its
  own, and they are poor. The Phase 7 headline's rule family needs a different mechanism — a
  structured, pathway-specific third factor (7a-ii) or decorrelating terms — before another
  panel is worth running.
- The one wiring-specific signal in four panels is reward-free Hebbian alignment holding or
  finding better fixed points on the real wiring than on its scramble. It is descriptive every
  time.
- Low initial noise is a free improvement to connectome PPO; a behavioural-clone warm start is
  not.

## Limitations

- n = 8, stated in advance; every reverse here clears its interval by a wide margin, so power
  was not the limit for the results that matter.
- One teacher, one rollout distribution, one cloning loss. A clone that matched the switch's
  sign rather than its mean might have started higher.
- The plastic-set clone's inflated norms are the state the rule started from; a clone at the
  construction norm was not run.
- Attempt 1 is a record, not a result; its logs are kept beside the analysis and not read.

## Next Steps

**7a-ii** with the third factor made structured (pathway-specific instruction through the
receptor atlas), which is where this rule family's next mechanism lives; the sparse random MLP
arm and a rule variant with anti-Hebbian/decorrelating terms stay queued behind it. Carry
forward: `initial_log_std −1.0` for any connectome PPO arm; a warm-started PPO arm needs a
critic warm-up before it is a fair comparator; and the Hebbian-holds-the-clone contrast is the
next candidate for a registered test if a statistic for a bimodal outcome is chosen in advance.

## Data References

- Registration and design: `openspec/changes/archive/2026-09-07-add-l4-warm-start-tooling/`,
  `openspec/changes/archive/2026-09-08-add-l4-warm-start-panel/`; capabilities
  `openspec/specs/l4-plasticity-panel/spec.md`, `openspec/specs/behavioural-cloning/spec.md`,
  `openspec/specs/weight-persistence/spec.md`.
- Everything the panel produced:
  [supporting/043-l4-warm-start/](supporting/043-l4-warm-start/details.md) — `launch.md`,
  `teacher.json` and the frozen recording config, `clones.json`, `clone-fit-notes.md`,
  `attempt1-plastic.md`, `panel.json`, `per-seed.csv`, `curves.csv`, `_manifest.txt`,
  `details.md`.
- Tooling: `brain/arch/connectome_ppo.py` (persistence), `brain/rollouts.py`,
  `scripts/campaigns/l4_behavioural_clone.py`, `scripts/campaigns/l4_warm_start_campaign.py`,
  `scripts/analysis/l4_warm_start.py`; `learning_rules/three_factor.py` (`reset_state`).
