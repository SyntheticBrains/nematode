# 125: Phase 10 — C.1 on Benchmark v4: It Lacks an On-Food Dwelling Reversal Rate; Given One, a Continued Fit Reaches v4 and the Box Centre Does Not; the Refit Is Deferred

**Status**: completed, as a diagnosis. No fit is registered. **C.1's 25-parameter model cannot reach benchmark v4**, and one mechanism closes the gap: a dwelling reversal rate of its own on food.

- With it, a fit continued from C.1 (v3)'s answer passes the wild type in 4 of 5 discarded populations (D 1.28–1.44).
- From the box centre, where a registered fit would start, no run reaches the passing basin: 0 of 5 in each of three.
- **The maintainer deferred the refit** (2026-11-05) until C.4 approaches, to run once on the benchmark current then.

> | Probe (one restart, 5 populations each) | Model | Start | Passing | D | What fails |
> |---|---|---|---|---|---|
> | readiness | 25 | C.1 (v3)'s answer | 0 | 1.39–1.46 | the on-lawn onset rate |
> | readiness | 25 | box centre | 0 | 1.78–1.86 | the on-lawn onset rate |
> | reversal–speed coupling | 26 | readiness answer | 0 | 1.56–1.72 | the on-lawn onset rate |
> | intermittent dwelling | 27 | readiness answer | 0 | 1.30–1.50 | the on-lawn onset rate |
> | a state-free off-food rate | 26 | readiness answer | 0 | 3.58–3.80 | local search's decay (γ undefined) |
> | **an on-food dwelling rate** | **26** | **readiness answer** | **4** | **1.28–1.44** | **one roaming-bout length** |
> | an on-food dwelling rate | 26 | box centre | 0 | 1.85–1.92 | the onset rate; its states swapped |
> | the same, states ordered | 26 | box centre | 0 | 2.87–3.02 | the kinematics |

**Date**: 2026-11-05.

**OpenSpec change**: `c1-v4-diagnosis`.

**Records**: [`supporting/125-c1-v4-fit/`](supporting/125-c1-v4-fit/):

- `probes/`: every probe, the state diagnosis and the two sensitivity checks, each with its script as run;
- `identity.json` (the model unchanged without the option);
- `parked_fit_design.md`: the drafted fit design, not registered.

## Objective

Benchmark v4 ([Logbook 124](124-benchmark-v4.md)) fails C.1's v3 answer in all 5 of its registered populations. Its on-lawn onset rate is 0.78–0.85 per minute against real worms' 2.28 (z −6.4 to −6.7). The maintainer chose to refit C.1 on v4 before C.2's next change, first asking whether its model can reach v4.

## Method

Every probe is one sep-CMA restart against v4's definedness-first ranking score at the band centres (n = 200), with C.1b's doubly guarded stop and a cap of 300, on discarded seeds. Each answer is graded on v4 with presence on 5 populations, and it **reaches** v4 when at least 3 pass. The probes ran on the commit that holds every candidate (`886e7bfc`), except the off-food rate's, whose code its script records. The eight took 10.75 h on 16 workers.

## The diagnosis

**The readiness probes** fail on the on-lawn onset rate alone, in every population: z −3.3 to −3.9 continued from C.1 (v3)'s answer, and −5.6 to −5.8 from the box centre. The pause share and the flip fraction pass.

**Where real worms' onsets come from** (`state_diagnosis.json`). Each on-lawn second is assigned its 10-s window's state, on PD1074's band cohorts and on the readiness answer's populations:

| On-lawn, by window state | Real | C.1 (readiness answer) |
|---|---|---|
| dwelling share of classified seconds | 83% | 74% |
| dwelling onsets per minute | **2.12** | **0.78** |
| dwelling 1-s steps below 0.02 mm, at least 0.03 mm | 55%, 31% | 63%, 20% |
| dwelling flips per minute | 1.26 | 1.35 |
| roaming onsets per minute | 3.09 | 3.36 |
| roaming flips per minute | 3.91 | 6.01 |

Real worms make most on-lawn onsets while dwelling. Their dwelling is intermittent: over half the seconds paused, yet a third of the steps clear the detector's 0.03 mm floor. C.1 dwells at a near-constant 0.013 mm/s with its dwelling reversal rate at its floor (0.001 per s), and puts its reversals into roaming (0.146 per s), where its flips overshoot.

**Why the fit does not raise it** (`dwell_sensitivity.json`, one population each on shared animals). On the intermittent-dwelling answer:

| Change | On-lawn onsets z | log early/late z | Chen flips z | dwelling speed p90 z |
|---|---|---|---|---|
| `r_dwell` 0.03 | −2.2 | −3.1 | +2.1 | +2.7 |
| `r_dwell` 0.1 | +0.1 | −3.8 | +4.5 | +6.6 |
| moving speed 0.05 mm/s | — | — | — | +6.5 (median +7.2) |

The same rate sets reversals off food, so raising it flattens López-Cruz's search curve and over-flips Chen's chemotaxis. Faster dwelling movement breaks the 10-s window speeds. **The conflict is on food against off food.**

## The candidates

- **A reversal–speed coupling** (v2's reachability worm's, the reversal hazard × exp(β z) on the window's speed draw): 0 of 5, the onset rate z −3.7 to −4.1, β 0.76. It acts on whole windows.
- **Intermittent dwelling** (a per-second pause chain on food): 0 of 5, the onset rate z −4.0 to −4.2. The fit paused 80% of dwelling seconds in 1.7-s pauses, moving at 0.028 mm/s, still below the floor, with `r_dwell` at its floor.
- **A state-free off-food reversal rate:** 0 of 5. It ran to the cap with one statistic undefined throughout: local search's γ, in every population, with its presence failing. C.1 makes its off-food decay from the two states' different rates, and one rate replaced them. Its on-lawn onsets did come within the bar (z −1.3 to −1.6), with `r_dwell` raised to 0.097.
- **An on-food dwelling reversal rate** (`r_dwell_on_food`), which keeps both state rates off food. In its sensitivity test (`ondwell_sensitivity.json`) it moves no off-food statistic. At 0.03 the on-lawn onsets pass (z −2.7), and the small lawn's roaming fraction and roaming bouts fail, which a fit retunes.

**The fit with it** (`mechanism_ondwell.json`), continued from the readiness answer, plateaued at generation 101. It passes 4 of 5 populations (D 1.28–1.44), and the fifth fails one small-lawn roaming-bout length (z +3.23).

- It reverses at 0.053 per s while dwelling on food, and at 0.0015 off it.
- The margins are thin: the onset rate sits at z −2.6 to −3.0 against the bar's 3.078, and the small lawn's roaming bouts at +2.2 to +3.2.
- Local search and chemotaxis are present in every population.
- Its reversal share over the off-food windows is 4.6–4.8%, against Roberts' 17.3%. That share is reported, not graded.

## The box centre

A registered fit starts from the box centre, so the 26-parameter model was probed from there with the stop tightened (every median in the last 40 generations below 100).

- **It swapped its states** (`box_ondwell.json`: 0 of 5, D 1.85–1.92). Its state 0 is the fast one (0.118 against 0.016 mm/s). `r_dwell_on_food` acts only on state 0, so here it acts on the fast state, and the slow state cannot reverse on food. The onset rate fails at z −4.1 to −4.6.
- **Ordering the states** (`ordered_box.json`, the dwelling speed a log-uniform fraction 0.025–1 of the roaming speed) removed the swap, and the run found another dead end: 0 of 5, D 2.87–3.02. Its roaming state sits at three bounds (speed spread 1.5, omega 0.5 per s, reversal 0.48 per s), and its kinematics miss widely: dwelling speed p90 z +10, median +7, roaming speed −5.

With Logbook 121's 25-parameter model, about one box-centre restart in three found the passing basin. Here none of three did.

## The decision

The maintainer chose to **defer the refit** and ship this diagnosis.

- **A warm start conflicts with D38 as amended,** which sets C.1's search box and x₀ "without search data". C.1 (v3)'s answer was fitted on every v3 family, local search and chemotaxis included. A continued fit would also be a local refinement, and its recovery would start next to its own truths.
- **Box-centre restarts have unmeasured odds,** and the recovery's own box-centre fits risk its gate.
- **Nothing downstream waits on it.** C.2, F.0b-ii and W do not depend on C.1's refit. The benchmark has had three versions since 2026-10-28 (v2–v4), two of them after a fit exposed a gap, so a refit now might be redone.

**The refit runs once, when C.4 approaches,** on the benchmark current then, from the parked design (`parked_fit_design.md`). Its start is decided at that point: a dated D38 amendment for a warm start, or a box-centre design with its odds measured first.

## What this establishes, and what it does not

- **C.1's model can reproduce v4's wild type, given one mechanism.** It answers a measured failure, the missing dwelling reversals on the lawn, and it moves no off-food statistic. The pass is at the threshold, from a continued start, on discarded populations. No registered verdict is read.
- **The mechanism was chosen by looking.** It is C.1's fourth mechanism chosen on the benchmark's families. Every candidate's record is committed, and C.1's mutant predictions remain its held-out test.
- **The box centre does not find the basin.** That is a property of the fit from the box centre, not of the model or the benchmark. So the roadmap's risk row ("C.1 fails the wild type: the primitives or the benchmark are wrong") is not triggered.

## Registered consequences

- **`BatchedMechanistic` gains `r_dwell_on_food`,** off by default and byte-identical without it (59 of 59 identity jobs). The coupling and intermittent dwelling are not kept.
- **C.4:** C.1 enters from its refit, run when C.4 approaches. Until then C.1 (v3) fails v4 (0 of 5), and C.2's yardstick is the continued probe's D (1.28–1.44), as description.
- **Next:** C.2's next change: the run mode, the staged procedure and the readiness restarts against v4.

## Artefacts

- `docs/experiments/logbooks/supporting/125-c1-v4-fit/probes/`: `readiness_warm.json`, `readiness_box.json`, `state_diagnosis.json`, `mechanism_coupling.json`, `mechanism_pauses.json`, `dwell_sensitivity.json`, `mechanism_offfood.json`, `ondwell_sensitivity.json`, `mechanism_ondwell.json`, `box_ondwell.json`, `ordered_box.json`, `identity_all_options.json`, and each script as run (`*.py.txt`).
- Seeds, all discarded: fits 620, 621, 628, 629, 633, 636, 640 and 644 M; their answers' gradings 625, 626, 630, 631, 635, 638, 642 and 646 M; the diagnosis 627 M; the sensitivity checks 632 M; smoke checks 99.62 M and 99.63 M; CMA 9441–9448.
