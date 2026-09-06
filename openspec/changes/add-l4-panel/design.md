# Design: the L4 2×2 panel

## Context

The panel resolves roadmap D2 under the D10 arm set on the frozen C3 substrate (Amendment A,
Logbook 038). The project's paired-seed statistics layer, the level-agnostic plateau metric, and the
parallel campaign runner all exist and are reused verbatim. What this change adds is discipline —
the order of operations — plus the small amount of tooling the discipline needs.

Facts the design rests on, measured before writing it:

- No plastic arm has run beyond the smoke test. A 40-episode timing run of the plastic wild-type
  and plastic MLP arms shows both at chance (5% and 12.5% full-clear), which says nothing about
  learning yet and everything about why a pilot must precede the budget pin.
- A plastic connectome episode costs roughly a fifth of a second early in training (40 episodes in
  8.9 s), cheaper than the PPO arm since there is no batched update. Even with episodes lengthening
  as the worm survives, a seven-arm × eight-seed panel at a 029-scale budget fits in an afternoon on
  16 workers. Compute does not constrain the design.
- The 034 controls harness already computes the primary contrast exactly as this panel needs it
  (`plateau_tail` + `paired_seed_wilcoxon_bootstrap` + `bh_fdr`). This change generalises that
  script's shape to seven arms and a pre-registered family; it does not touch the statistics.

## Goals / Non-Goals

**Goals**

- Fix the success tests, the confirmatory family, and the verdict map **before** the pilot runs.
- Pin the shared rule recipe and the uniform budget from a pilot on seeds the panel never uses, by
  rules stated in advance, then freeze both.
- Run the panel once, analyse it with a committed harness, persist the data for the logbook.
- Keep the robustness branch cheap and honest: its one permitted sensitivity pass is defined now and
  cannot change the verdict.

**Non-Goals**

- The logbook and its narrative (A.8).
- Any change to the substrate, the rule's arithmetic, or the trace mechanism.
- Comparing the plastic arms to the PPO-trained arms quantitatively: the commensurability rule
  forbids it. Logbook 029's table enters the logbook as descriptive frame only.
- Tuning anything per arm. One recipe, every arm.

## Decisions

### D1. Arms: the full 2×3 factorial plus the yardstick — seven arms

| arm key | config | wiring | rule |
|---|---|---|---|
| `wt_frozen` | `…_plastic_frozen.yml` | wild-type | frozen (three-factor, `freeze_updates`) |
| `wt_hebbian` | `…_plastic_hebbian.yml` | wild-type | unmodulated Hebbian |
| `wt_plastic` | `…_plastic.yml` | wild-type | three-factor |
| `rn_frozen` | `…_plastic_frozen_rewired_null.yml` (new) | rewired-null | frozen |
| `rn_hebbian` | `…_plastic_hebbian_rewired_null.yml` (new) | rewired-null | unmodulated Hebbian |
| `rn_plastic` | `…_plastic_rewired_null.yml` | rewired-null | three-factor |
| `mlp_plastic` | `mlpppo_…_plastic.yml` | dense MLP | three-factor |

All connectome configs are under `configs/scenarios/foraging_predator_thermal/` with the
`connectomeppo_small_continuous2d_combined_klinotaxis` prefix. The two new configs are one `wiring`
key off their wild-type parents, following the derived-suffix naming convention so the parent name
stays a prefix. Ratified with Chris: the rewired floors are worth their 16 runs because they make the
learning-gain contrast (D4, T4) available; without `rn_frozen` a difference between the two plastic
arms could be an initial-policy offset of the random wiring rather than anything the rule did.
The two rewired floors share `rn_plastic`'s exact wiring seed for seed (`rewire_seed` derives from
the run seed in every rewired config), which is what makes T4's within-wiring gain a like-for-like
subtraction; task 1.3 pins it with a mask-equality test.

### D2. Seeds: panel seeds 1–8, pilot seeds 101–102, never mixed

The panel uses seeds 1–8, paired across every arm, matching Logbooks 029 and 034 so the connectome's
seed-to-seed variance is the same population those logbooks documented. The pilot uses seeds 101 and
102 only. A recipe or budget chosen on seeds the panel then runs would let the pilot's luck leak into
the confirmatory result; disjoint seeds make the pilot strictly pre-flight. `rewire_seed` stays unset
everywhere so the rewired arms pair with the wild-type arms seed for seed.

### D3. Ranked metric and convergence: reused verbatim

The per-seed ranked metric is the committed plateau-tail (final-quarter) full-clear success from the
run's log, exactly as 029 and 034. The per-seed convergence verdict is the level-agnostic detector's
`convergence_run`, read from the experiment JSON the run writes under `experiments/<id>/` when
`--track-experiment` is passed through the campaign runner; the harness locates that JSON from the
experiment id the run logs. A run whose JSON is missing is reported with convergence unknown, never
silently as converged. Learning curves (rolling full-clear per 250 episodes) are exported per seed
for the logbook's descriptive sample-efficiency read; they are not tested.

### D4. The pre-registered tests, the BH-FDR family, and the MLP band

Every confirmatory test is a paired per-seed delta on the ranked metric, one-sided Wilcoxon
signed-rank, 80% bootstrap CI (1000 resamples, seeded RNG 42), through the committed helper.

**Confirmatory family** — four one-sided p-values corrected together by BH-FDR at α = 0.05:

| id | test | direction | reads |
|---|---|---|---|
| **T1** | `wt_plastic` vs `rn_plastic` | wild-type > rewired | D2 test (i), the primary contrast |
| **T2** | `wt_plastic` vs `wt_frozen` | plastic > frozen | sanity floor: something was learned |
| **T3** | `wt_plastic` vs `wt_hebbian` | plastic > Hebbian | sanity floor: learned *from reward* |
| **T4** | (`wt_plastic` − `wt_frozen`) vs (`rn_plastic` − `rn_frozen`) | wild-type gain > rewired gain | primary contrast on learning gain |

A test passes when its BH-FDR q < 0.05 **and** its mean delta is positive.

**MLP band test (D2 test ii)** — CI-based, outside the family (it has no one-sided p to correct):
the 80% bootstrap CI of the paired delta `wt_plastic − mlp_plastic` **contains or lies above zero →
PASS**; entirely below zero → FAIL. Ratified with Chris over a "mean within ±1 sd" reading: it is
paired, seeded, reuses the committed helper, and is the same reading 034 used for its verdict.
Its asymmetry is stated here so the logbook cannot miss it: containing zero is the *null* outcome,
so a wide interval from a high-variance arm passes by noise. The band test is therefore the weaker
of the two D2 tests by construction, a `recovery` verdict rests on T1, and the harness reports the
band delta's mean and interval width beside the outcome so a noise pass is never presented as
parity.

**Direction-agnostic reporting.** The helper's Wilcoxon is one-sided, so a significant *reverse*
result cannot show up in q. As in 034, a reverse result is detected by the CI lying entirely on the
wrong side of zero and is reported as its own named outcome, never folded into "not significant".

**Descriptive, uncorrected**: all 21 pairwise deltas among the seven arms (Wilcoxon p + CI), the
per-behaviour sub-metrics (foods, evasion rate, thermal comfort), the converged fraction per arm, and
the per-seed values. The family is held to the four pre-registered tests so the correction stays
interpretable, per the comparison-protocol spec.

**Ensemble-invariance (roadmap bar (a))**: the harness reports, for T1 and T4, the number of the
eight paired seeds whose delta is positive. The default claim type for every panel result is a
*performance claim*; a dynamics claim is admissible only if the T1 delta is positive on at least
seven of eight seeds and bar (b) is separately met. This is a report, not a test; it changes no
verdict.

### D5. The verdict map

Evaluated in this order, from the family results:

| verdict | condition | roadmap branch |
|---|---|---|
| `sanity_floor_fail` | T2 or T3 fails | "L4 plasticity fails to beat its baselines" |
| `rewired_beats_wild_type` | floors pass; T1's CI lies entirely below zero | reported as its own outcome (034 precedent) |
| `recovery` | floors pass; T1 passes; band test PASS | GO branch, D2 (i) and (ii) both met |
| `structure_only` | floors pass; T1 passes; band test FAIL | the roadmap's "strong even if it trails the MLP" outcome |
| `robustness` | floors pass; T1 fails; CI spans zero | "Partial D2 outcome", the pre-registered negative-result closure |
| `inconclusive` | floors pass; T1 fails; CI does not span zero and is not entirely below | reported honestly; no branch claimed |

T4 annotates the verdict (agrees / disagrees with T1 on the gain read) and never changes it. The
band test is reported under every verdict.

### D6. The pilot: one grid, neutral selection, pre-registered budget rule

- **Grid**: `plasticity_rate ∈ {0.003, 0.01, 0.03}` — the shipped default and a third and three
  times it. Every other rule hyperparameter (weight decay, bound, baseline rate, trace decay) stays
  at its default; the grid is one axis on purpose, so the pilot is a recipe check, not a search.
- **Runs**: the five rule-bearing arms (the wild-type and rewired Hebbian and three-factor connectome
  arms, and the MLP) at each grid point, plus the frozen arms once (their weights
  never move, so the rate is irrelevant to them), at pilot seeds 101–102, at a pilot budget of
  3000 episodes. If any three-factor arm at the selected rate has not converged on either pilot
  seed by 3000, that arm's pilot is extended once to 6000 before the budget is pinned (the
  protocol's extend-and-rerun). If it still has no plateau on both pilot seeds at 6000, the budget is
  pinned at 6000, the arm is flagged non-converging in the pilot summary, and the panel runs as
  registered. The frozen arms run in the pilot as a descriptive floor read only; they feed neither
  the selection nor the budget.
- **Recipe selection**: the rate that maximises the **pooled mean** plateau-tail success of the
  three three-factor arms (`wt_plastic`, `rn_plastic`, `mlp_plastic`) over the pilot seeds. Ties
  go to the default. Pooling is the neutral choice: selecting on `wt_plastic` alone would hand the
  arm the hypothesis favours its best-case recipe and make T1 and the band test anti-conservative.
  The selected rate is written explicitly into all **seven** plastic-family configs — the six
  connectome arms and the MLP; leaving the MLP at the default would make "matched rule" false —
  even when it is the default, so the recipe is visible in the file and not inherited silently.
- **Budget rule**: the panel's uniform budget is the smallest multiple of 500 that is at least
  1.25 × the latest convergence onset among converged pilot runs of any arm at the selected rate,
  and never below 2000. At panel time, a seed still climbing at that budget gets exactly one
  extension to 1.5 × budget (the 029 top-up pattern, pre-registered here rather than improvised); a
  seed that still has no plateau is flagged and ranked on its plateau-tail per the protocol. An
  extension, in the pilot or the panel, is a **fresh run at the longer budget at the same seed**:
  the connectome brain has no weight persistence, so nothing can resume. Runs are
  seed-deterministic, so the longer run's first episodes replay the shorter run's exactly and it is
  a true continuation; its log replaces the shorter one in the manifest, and the launch record
  lists every extension. A per-arm extension is a separate campaign invocation, since a campaign's
  episode count is uniform across its configs.
- **Pinning**: both numbers land in this design as a dated amendment (§ Pinned values, below) with
  the pilot's summary JSON committed under the supporting directory, before the panel launches.

### D7. Launch record before analysis

Before the panel command runs, a `launch.md` under the supporting directory records the commit SHA,
the exact command, the seed list, the budget, and the recipe. It is committed before any panel log
is read. The harness output is deterministic given the logs, so the analysis can be re-run by
anyone from the committed data.

### D8. The robustness branch's one sensitivity pass, defined now

If the verdict is `robustness`, the primary pair (`wt_plastic`, `rn_plastic`) is re-run on the panel
seeds at the two grid rates the pilot did not select — 32 runs — and T1 is recomputed at each,
reported descriptively. This is the single sensitivity pass the roadmap permits. It informs the
logbook whether the null is recipe-robust; it **cannot** change the verdict. No other re-run is
permitted under this change.

### D9. What may change after the pilot, and what may not

May change (by the pilot, through the rules in D6, recorded by dated amendment): the recipe's
`plasticity_rate`, the uniform budget. May not change: the arms, the seeds, the metric, the tests,
the family, the verdict map, the band rule, the sensitivity pass. If the pilot reveals that the
protocol itself is broken (for example, the rule never leaves chance at any grid rate), the panel
still runs as registered — that outcome is `sanity_floor_fail`, which the roadmap already names as
publishable — and any redesign is a new change with its own pre-registration.

### D10. Harness and pilot-runner shape

- `scripts/analysis/l4_panel.py`: `--campaign-dir <dir>` (reads `logs/*.log`, maps each config stem
  to its arm key through a fixed registry, parses the seed from the label; in confirmatory mode any seed outside 1–8 is rejected, so a pilot log cannot enter a test) or `--manifest` (`<arm> <seed> <log>` lines, the 034 format); `--out panel.json`; `--csv per-seed.csv`; `--curves curves.csv`. `--pilot` switches to the grid summary: per rate per arm plateau-tail and convergence
  onset over the pilot seeds, the pooled selection, and the budget rule's output.
- `scripts/campaigns/l4_panel_pilot.py`: derives the grid configs from the committed arms (parent
  YAML plus one `plasticity_rate` key, written under `<out>/configs/` so the pilot is reproducible
  from its own directory) and invokes the campaign runner with `--track-experiment` and the headless
  theme passed through.
- Both are tested on synthetic logs and JSONs: the registry, the seed parse, each test's direction,
  the family size, every verdict row of the map, the band rule at both outcomes, the reverse-result
  detection, the pooled selection with a tie, the budget rule's rounding and floor, the seed guard, and the derived
  configs' one-key property.

## Risks / Trade-offs

- **The rule may not learn at any grid rate.** Then the panel reports `sanity_floor_fail` and Phase
  7 has its first citable negative result at the cost of an afternoon. The design forbids widening
  the grid post hoc; a second grid is a new change.
- **3000 pilot episodes may under-budget a slow learner.** The one extension to 6000 is the
  pre-registered remedy; the plateau metric normalises across budgets.
- **The Hebbian floor may beat the frozen floor by learning something reward-free** (the wiring's
  correlation structure). That is exactly why T3 exists alongside T2; a `wt_plastic` that clears T2
  but not T3 fails the floors, honestly.
- **The band test can pass on noise.** Containing zero is its null outcome (D4). The logbook must
  report the band delta's mean and interval width with any PASS, and a `recovery` verdict is read
  as resting on T1.
- **Seven arms invite 21 comparisons.** Only four are confirmatory; the rest are labelled descriptive
  in the JSON and the CSV so the logbook cannot quietly promote one.

## Open Questions

None that this change resolves post hoc. The pinned values below are filled by the pilot.

## Pilot 1 outcome (amendment dated 2026-09-06)

The pilot ran as registered and pinned nothing. The grid `{0.003, 0.01, 0.03}` is two orders of
magnitude too hot for this cell's reward scale: the terminal prediction error is about −10, so one
update moves weights by order 1 against an initialisation scale near 0.3, and the connectome arms
random-walk into the bound (the Hebbian floor ends 96% clamped). The MLP dies in episode 1 at 0.01
and is effectively frozen at 3e-3 and below; its per-weight trace is about a thousand times
smaller than the connectome's, so no single rate serves both substrates — a matched rate is not a
matched rule. Diagnostic probes show the connectome learning at 1e-4. Full record:
`docs/experiments/logbooks/supporting/040-l4-panel/pilot-1-notes.md`.

**Ratified with Chris:** the rule's scaling is made substrate-invariant in its own change before
the panel (a rule change is out of this change's scope and must be pre-registered on its own);
the grid is then re-registered here by dated amendment and the pilot re-run on the same pilot
seeds. Arms, panel seeds, metric, tests, family, verdict map, band rule and sensitivity pass are
unchanged. D9 anticipated this branch ("the rule never leaves chance at any grid rate") and
routed it to a new change rather than to a wider grid here; what the pilot added is that the
defect is scale, not the rule's ability to learn.

## Grid re-registration (amendment dated 2026-09-06)

Between pilot 1 and this amendment the rule changed twice, each in its own pre-registered
change: substrate-invariant scaling (a bounded modulator `tanh(δ / σ)` and a per-tensor trace
normalisation `E / ρ`), then centring of the compressed modulator (`tanh(δ / σ) − c`) after the
first probe with the switches on showed the uncentred form carries a positive mean on this
cell's skewed reward stream and drives a reward-blind Hebbian drift — the connectome collapsed
below its frozen floor and the MLP's activations exploded. Records:
`supporting/040-l4-panel/probe-2-uncentred-normalisation.md` and
`supporting/040-l4-panel/probe-3-centred-grid.md`.

**What changes in the registration, and only this:**

- Every arm config runs with both scaling switches on (`plasticity_normalise_modulator`,
  `plasticity_normalise_trace`); the plastic wild-type arm is therefore a four-key delta from its
  PPO parent, and every floor and rewired arm inherits all four. "One recipe, every arm" holds.
- The pilot grid is `plasticity_rate ∈ {3e-4, 1e-3, 3e-3}` with ties to `1e-3`, in the
  normalised units the rate now has (the root-mean-square Hebbian step per unit modulator).
  Ratified with Chris over a lower grid `{1e-4, 3e-4, 1e-3}` and a higher one
  `{1e-3, 3e-3, 1e-2}`: probe 3 shows both connectome wirings learning above the frozen floor at
  every rate, with saturation appearing only at `3e-3` (14–17% of synapses on the bound by
  episode 600), which is also the strongest learner at that horizon. The grid brackets it with a
  rate that learns without touching the bound, and the pilot's 3000-episode plateau-tail selection
  decides whether `3e-3` survives its saturation.

Arms, panel seeds, pilot seeds, pilot budget, metric, tests, family, verdict map, band rule,
extension rules and sensitivity pass are unchanged. One reading is recorded in advance: in
probe 3 the MLP yardstick learns at no rate and its trace scale still grows (`1e2–1e5` from
`0.01`), a property of a dense ReLU stack under a local Hebbian rule rather than an artefact. If
that holds in the pilot, D2 test (ii) passes by construction, and the band test's stated
asymmetry (D4) is what the logbook must say about it.

## Robustness probes and their rules (amendment dated 2026-09-06, written before the probes ran)

Pilot 2 ran on the re-registered grid and its rules selected `1e-3` and a budget of 2000, but the
pin was withheld (record: `supporting/040-l4-panel/pilot-2-notes.md`): the plastic arms sat near
their frozen floors while a fifth of the synapses were clamped on the bound, and the MLP yardstick
exploded or died. Three defaults of our own rule and arms explained it and were fixed in the
rule-robustness change: every plastic arm explored at action std 1.0 forever, the decay could not
hold a coherent Hebbian drive, and the yardstick's ReLU units are unbounded. The mechanisms are
default-off; the panel turns them on and chooses their values here, by the rules below, stated
before any probe result was read.

**Probe 4** (diagnostic, seed 101, 600 episodes, rate `1e-3`, both scaling switches on): every arm
with `plasticity_homeostasis: true`, the MLP arm with `activation: tanh`, and `initial_log_std ∈ {0, −0.5, −1.0, −1.5}` on the three three-factor arms and both frozen floors (the floors move with
the noise too, so the paired read must see them).

- **Homeostasis is pinned on for every arm** by decision, not by the probe: it is the runaway
  control the design adopted (rule-robustness D2). The probe confirms it by reading the
  saturated fraction, expected near zero.
- **The MLP arm is pinned to `tanh`** by decision, for the same reason (rule-robustness D3). The
  probe reads its trace scale, expected bounded, and whether it learns at all.
- **`initial_log_std` is selected by the probe**: the value maximising the pooled mean
  plateau-tail success (final quarter of 600 episodes) of the three three-factor arms, the same
  pooled principle the rate uses; ties go to the value nearest zero, the historical default. It is
  written into all seven arm configs.

Pilot 3 then runs the registered grid `{3e-4, 1e-3, 3e-3}` on seeds 101–102 with these values,
and the pin of recipe and budget follows the registered rules unchanged. Arms, panel seeds,
metric, tests, family, verdict map, band rule, extensions and sensitivity pass are unchanged.

## Pinned values (filled by dated amendment before launch)

- **Robustness values (pinned 2026-09-06 from probe 4,
  `supporting/040-l4-panel/probe-4-robustness.md`)**: `plasticity_homeostasis: true` and
  `initial_log_std: -1.0` on all seven arms (the pooled rule selected −1.0: pooled plateau-tail
  35.1 against 7.8, 8.7 and 4.4 at 0, −0.5 and −1.5); `activation: tanh` on the MLP arm. The
  plastic wild-type arm is therefore a six-key delta from its PPO parent and the MLP arm a
  seven-key one; every floor and rewired arm inherits its parent's keys.
- **Recipe** (`plasticity_rate`): **`0.001`**, pinned 2026-09-06 from pilot 3 by the registered
  pooled rule (pooled plateau-tail 16.3 against 8.1 at 3e-4 and 11.8 at 3e-3), written into all
  seven arm configs. Record: `supporting/040-l4-panel/pilot-3-notes.md`, summary `pilot.json`.
- **Uniform budget** (episodes): **3000**, pinned 2026-09-06 by the registered rule (latest
  converged onset at the selected rate 2090, × 1.25 = 2612, rounded up to the next 500). The
  single pre-registered extension for a seed still climbing at 3000 is a fresh run at 4500.
- **Pilot summary**: `docs/experiments/logbooks/supporting/040-l4-panel/pilot.json` (pilot 3).
