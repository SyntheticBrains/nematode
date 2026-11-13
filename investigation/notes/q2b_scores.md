# Q2 (second half): how C.1 and C.2 score on the Phase 10 benchmark

Paths are relative to the worktree root
`/Users/chriszaharia/projects/personal/nematode/.claude/worktrees/fit-first-investigation` unless
prefixed `feat/fit-c2:` (read with `git -C /Users/chriszaharia/projects/personal/nematode show feat/fit-c2:<path>`).
"LB n" means Logbook n. Line numbers are those of the logbook file. Read-only audit. Nothing was run
except small JSON spot-checks.

## 0. Reading conventions (needed to interpret every score)

- **A population** is 200 freshly simulated animals. **A population passes the wild type** when every
  graded statistic sits at |z| ≤ z\* and the presence tests (local search, and chemotaxis from v2 on)
  pass. z is measured on the bands' between-date scale at n = 200 (LB 124:34–36).
- **The verdict** for a controller is normally "the wild type passes in ≥ 3 of 5 registered
  populations" (LB 113:13; LB 119:50; LB 121:21; feat/fit-c2:.../133-c2-fit/launch.md:134).
- **z\*** is 2.84 (v1, 11 statistics), 3.0078 (v2, 19), 3.0381 (v3, 21) and 3.0781 (v4, 24). The mutant
  bar is 2.24 (LB 113:53; LB 119:50; LB 124:15; encodings.md:7).
- **D** is the weighted RMS of capped z over all families. **D_fit** covers only the families C.2 was
  originally fitted on: the small lawn and uniform well, plus kinematics from v2 and activity from v4
  (LB 115:63–71; LB 124:35).
- **The mutants** are graded on very few statistics: Scheer's small-lawn roaming fraction (tph-1,
  tax-4 by direction) and leaving. **pdfr-1 is graded by magnitude only** (its direction is not
  resolved), so a knockout that does nothing passes it (encodings.md:9–13; LB 128:12, 101).
- **Registered vs probe.**
  - *Registered verdicts* (launch.md committed before the run): LB 113, 119 and 121 for C.1, and the
    benchmark validity readings LB 115, 120 and 124. The slow-noise positive control is registered
    for C.2 (failed in LB 123; passed under launch 133's rule).
  - *Every other C.2 number below (LB 122–132) is description*: one restart, discarded seeds,
    constraints often applied in probe scripts. The logbooks say so, e.g. LB 126:188, LB 129:134,
    LB 131:3, 204.
  - C.2's registered fit (launch 133) has **not produced a result yet**.

## 1. C.1 (BatchedMechanistic), the mechanistic baseline

| Logbook | Benchmark | Model / fit | Verdict | D | Failing statistics | Status |
|---|---|---|---|---|---|---|
| 111 | (v1 statistics) | NPE on 20 params | informative gate fails (5.32 vs ≤ 1.42); same-parameter noise floor 4.08 > bar, so the gate was unreachable (LB 111:9–15) |: |: | registered, failed; NPE dropped |
| 112 | v1 | sep-CMA recovery of 4 synthetic truths | median gain 0.991 ≥ 0.75, pass (LB 112:3–15) |: |: | registered |
| 113 | **v1** | 20 params, 8 box-centre restarts, pop 16, plateau, cap 210, objective D on all 11 stats **incl. local search** (LB 113:34–36; launch.md:15 "Local search is fitted") | **3 of 5, pass at the threshold** (LB 113:13) | 0.95–1.26 | small-lawn roaming bout (z 3.55, 4.68) in the two failing populations (LB 113:58) | registered |
| 115 | v2 | v1 answer re-graded | fails 5/5 (LB 115:28) | 6.1–6.2 | (v2 found it passed v1 "partly through routes no real worm takes": 2.6× real roaming speed, etc., LB 115:45–50) | registered |
| 116/117 | v2 | recovery re-run | 0.690 fail; then 0.873 pass with the definedness-first ranking score, 4 restarts (AGENTS.md; memory) |: |: | registered |
| 119 | **v2** | 20 params, 8 box-centre restarts, ranking score, σ-guarded plateau, cap 248, selection on 3 separate populations; objective on all 4 assays / 5 families incl. chemotaxis (launch.md:17–18) | **0 of 5, fail** (LB 119:5, 72) | 1.71–1.87 | **chemotaxis presence in all 5** (drift −0.021 mm/s, z −2.9 to −3.1); pre-exit enrichment pop 0 (−3.12); exits/h pops 3–4 (−3.16). In pops 1–2 every one of 19 statistics passes and only presence fails (LB 119:73–75) | registered |
| 119 addendum | v2 | diagnosis probes | the fits are reversal-dominated off food: 52–56% of seconds reversing vs real 12–17%. v2's local search graded shape only, so a flat 9.5 reorientations per bin passed it (LB 119:163–192) |: |: | description; **the original diagnosis was retracted** (LB 119:3) |
| 120 | v3 | C.1b answers re-graded | each caught 5/5 (LB 120:10, 22) |: |: | registered |
| 121 | **v3** | **25 params** (+speed spread ×2, `v_off_food`, `sigma_off_food`, `kappa_off_food`), chosen by 7 probes on all five families (LB 121:40–67); recovery 0.985; 12 box-centre restarts, cap 330, selection on 5 | **3 of 5, pass at the threshold** (LB 121:3, 21). Restarts 4 (5/5) and 0 (4/5) also pass (LB 121:108–114) | 1.03–1.14 | small-lawn roaming bout (z +2.29 to +3.78 vs 3.04 [≈3.0381]); uniform-well roaming bout short (−1.52 to −2.69) (LB 121:123) | registered |
| 124 | v4 | v3 answer re-graded | **0 of 5** (LB 124:15) |: | on-lawn onset rate 0.78–0.85/min vs 2.28 (z −6.4 to −6.7) | registered |
| 125 | v4 | probes: +`r_dwell_on_food` (26 params) | continued from the v3 answer: 4 of 5 discarded pops; from the box centre: 0 of 5 in each of three runs (LB 125:9–18). **Refit deferred to C.4** (LB 125:90–96) | 1.28–1.44 (warm) | one roaming-bout length (+3.23) (LB 125:72) | description only |

Fit spot-checks: `supporting/113-c1-fit/fit.json` (chosen 3, 3 populations, diagnosis
`small_lawn/roaming_bout_s`); `119-c1b-fit/fit.json` (chosen 2, 0 populations; chemotaxis drift and
presence, exits, pre-exit); `121-c1-v3-fit/fit.json` (chosen 6, 3 populations, D 1.03–1.14; restarts 0, 4
and 6 pass). All match the logbooks.

**C.1's held-out families:**

- **Local search and chemotaxis are fitted for C.1** from the start (LB 113 launch:15; LB 115:56), so
  they are never held out for C.1.
- **Mutants:**
  - The mutants were C.1's declared held-out test in C.4 (LB 121:67, 131; LB 125:101).
  - They have **never been graded for C.1** (LB 113:96, an open task). pdfr-1 is not encodable in C.1
    (encodings.md:20).
  - **The D37 amendment of 2026-11-10 makes C.1's refit before C.4 also fit tph-1 and tax-4**
    (feat/fit-c2 roadmap D37 amendment). This removes the held-out status the earlier logbooks relied
    on.
- **Reversal share** (reported, never graded): about 3% off food at v3 (LB 121:132) and 4.6–4.8% at
  v4's warm probe (LB 125:77), against Roberts' 17.3%. C.1 under-reverses and still passes.
  **"v3 does not constrain reversal from below"** (LB 121:132).
- **The persistent miss** is the same at v1, v3 and v4: one roaming state cannot fit the roaming bouts
  of both of Scheer's lawns (LB 113:73; LB 121:133).

## 2. C.2 (compact connectome encoder)

The parameter count varies by setting: 196 by default, 132 per-ligand, 197 with OU speed noise, and
137 under the procedure (per-ligand + timescale + 4 pool centres) (LB 122:55–64; LB 130:51).

| Logbook | Benchmark / objective | What was fitted | Wild type | D / D_fit | Failing statistics; held-out families | Status |
|---|---|---|---|---|---|---|
| 122 | v3 D_fit (lawns + kinematics) | box-centre fits stalled at score ≈10.6 (time on lawn z −116). Box fixed twice (wiring range [0, 1.5]; fixed-sign strengths log-scaled). Staged fit: 18 params × 150 gens, then all 196 × 500 gens at σ₀ 0.1 (LB 122:124–155) | not graded on 5 populations; readiness "not run" (LB 122:32) | score 3.43, D_fit 3.5–3.6, still improving at the cap | dwelling p90 +6.5, roaming p90 −5.3, uniform-well roaming bout −5.1, small-lawn roaming fraction +5.3, dwelling bout −3.4, from one re-simulation (LB 122:157–165). **Fitted before the López-Cruz sign correction** (LB 122:222) | description |
| 123 | same answer | slow OU speed noise added |: |: | the answer **does not run**: speed pinned at 0, curvature at max, moves only by reversals and delta turns (LB 123:5–13). **Mutants all fail** (tph-1 z −5.3; pdfr-1 z +4.4; tax-4 wrong direction, capped at 10) (LB 123:52–58). Held-out off-food: onset z +5.9, flips +6.1 (LB 123:131). Registered slow-noise control **fails** (LB 123:31) | control registered; rest description |
| 124 | v4 |: | stage-2 answer caught 5/5 by the new activity family (LB 124:9, 113) |: | flip fraction z −10, onset z −9.7, pause +3.5 | registered (benchmark validity) |
| 126 | v4 D_fit (4 families) | 7 designs stall at score 8.6–9.6. The winner: an **imitation of C.1's v4 warm-probe answer** (gradient descent, R² ≈ 0.3), then sep-CMA, then a **responsiveness guard + population 48**, 250 gens (LB 126:19–28, 97–118). Bounds and guards applied in probe scripts (LB 126:37) | **0 of 5** (LB 126:5, 144) | D_fit 1.29–1.40 (every fitted family passes 5/5); D 3.5–3.7 | **local search fails 5/5** (log early/late ≈ 0, z −3.7, presence fails). **Chemotaxis fails 5/5** (drift 0.004 vs 0.026; weathervane z +4.3–4.5; onset +3.9; flips +5.9; presence). **tph-1 fails 5/5** (z −5.2); **tax-4 fails 5/5** (wrong direction); pdfr-1 passes 4/5 (LB 126:124–144). The no-imitation ablation reaches D_fit 2.15, 2 of 4 families, at gen 250 (LB 126:148–168) | description |
| 127 | v4, **all six families now in the objective** (held-out local search and chemotaxis moved into the fit after they failed in 126) | 136 params, pop 48, σ₀ 0.1, from 126's answer, 150 gens (LB 127:40–45) | 5 of 6 families on **one** population (LB 127:47, 58) | D 1.77, D_fit 1.69 | chemotaxis: flip fraction z +4.1 and presence. tph-1 z −4.9 fails; **pdfr-1 z +3.4, wrong way**; tax-4 passes; mgl-1 inert (1.01× vs real 0.31×) (LB 127:67–72). **Every fit drives the inhibitory scale to 0.005–0.09** (LB 127:81–91) | description |
| 128 | v4 six families ± mutants | constrained V1/V2; leave-one-out; 60-gen fit with mutants as 3 of 9 families |: | D 3.37 → 2.48 | mutants trade against the wild type. tph-1 lost; tax-4 passes at gens 20–40, then lost; only pdfr-1's magnitude pass remains (LB 128:13, 111–120). PDF and ser-5 signs found reversed by the fit; set from receptor biology (LB 128:27–47) | description; pre-fixed branch not met |
| 129 | v4 six families + **3 mutants at half the objective** | from 128's answer (which descends from the imitation of C.1); inhibition ≥ 0.09; PDF and ser-5 excitatory; pop 48, σ₀ 0.05, 150 gens, 12.5 h (LB 129:33–36) | (a) **0 of 5**: roaming bout +3.57–3.62 (in 3), roaming p90 −3.08 to −3.21 (in 4); D 1.50–1.58. (b) +OU noise continuation: **0 of 5**: on-lawn onset −3.12 to −3.15; D 1.58–1.71. (c) **CEP food-contact gain +0.8 set by hand**, chosen among ~10 nudges on one population (LB 129:96–110, 131): **3 of 5** (LB 129:114–122) | (c) D 1.63–1.68 | (c) pop 1: flip fraction +3.11; pop 2: roaming survival −3.30. Mutants 5/5 each and presence 5/5 in all three. "The mutants were fitted, not predicted" (LB 129:132) | description |
| 130 | same | procedure built into the package; answer reproduced (Δ ≤ 7.2e-7) | same 3 of 5 (LB 130:88–98) | within 0.01 | same | engineering |
| 131 | v4 six + mutants | **clean start, no C.1 imitation.** Graded guard; curriculum 1 (77 params), 2a (4 families, 300 gens; **cap raised 150 → 300 mid-run**), 2b (6 families, 150), 3 (+ mutants, 150) (LB 131:58–66). Then hinge, hinge2, hinge3 (failure-probability term), 50 gens each (LB 131:85–129) | curriculum 0/5 (onset −3.37 to −3.65); hinge 2/5; hinge2 3/5 (9/15); hinge3 3/5 (8/15) (LB 131:10–16) |: | frontier between on-lawn onset rate and dwelling p90 | description |
| 131 (refit) | same, **with a speed-following on-food reversal patched into the evaluator** | 50 gens from hinge3 under the coupled body (LB 131:175–186) | **5 of 5; 15 of 15** with 10 more (LB 131:184–185) | D 1.37–1.51 (10 pops) | no failures. Closest: flips +2.85 (SD 0.05), onset −2.64, log early/late −2.27, exits −2.21, dwelling p90 +2.10 (LB 131:190–196). Mutants 15/15 | description; "route adapted after each reading" (LB 131:204) |
| 132 | same | the option `follows_run_speed` with held-out-cohort knots | **10 of 10** fresh pops, as under the patch (LB 132:121–126) |: |: | confirmation, pre-fixed rule ≥ 8/10 |
| 133 | registered (below) | | **no result yet** | | | registered, pilot stage |

Spot-checks:

- `126-c2-run-mode/probes/full_grade.json`: populations_passing 0, D 3.54–3.66.
- `129-.../probes/grade_nudged.json`: wild_type_passing 3, mutants 5/5/5.
- `131-.../probes/hinge4.json`: verdict "wild type 5/5 … passes", check wild_type_pass_share 1.0 over
  10 pops, D 1.37–1.51.
- `132-.../confirmation.json`: wild_type_passes 10, rule met.

All match. Not verified: the off-food reversal share of 131's final answer (not in hinge4.json).
LB 129 reports 5.5–7% for its answers (LB 129:56, 88, 123), against Roberts' 17.3%.

### Launch record 133 (feat/fit-c2, 5 commits ahead of main, 2026-11-10 to 11-13)

- **Arm:** the encoded connectome, 137 params, under the procedure settings, on
  `connectome_c2_following`, i.e. with `follows_run_speed` (launch.md:13–20).
- **Objective:**
  - the ranking score on each stage's families;
  - the mutants at half the objective;
  - a saturation guard (+500), a graded responsiveness guard (+300·(1 − s/0.5)), and a **lawn guard**
    (added 2026-11-13);
  - in the final stage, 10·Σ Φ((|z| − bar)/SD), each statistic's chance of failing (launch.md:24–33).
- **Stages:** 1 (77 params, 64 animals, pop 24, 150 gens) → 2a (137, D_fit, 300) → 2b (six families
  - mutants, 200) → noise step (30 pops) → final (50). Fixed budgets; the plateau is reported only
    (launch.md:45–56).
- **Restarts:** up to six starts of the first half. A start is kept if its mean time on the lawn at
  2a gens 40/50/60 is ≥ 0.95 (launch.md:58–69).
- **Order:** readiness pilot (go: WT ≥ 3/5, tph-1 and tax-4 each ≥ 3/5) → F.0b-ii at **one truth**
  (the pilot's answer; gate gain ≥ 0.75; "cannot tell recovering it from fitting the wild type",
  launch.md:124) → registered fit → tax-4 leave-one-out as description (launch.md:82–179).
- **Verdict:** WT ≥ 3/5 and tph-1, tax-4 each ≥ 3/5; pdfr-1 reported only. **"The verdict is
  in-sample: every graded statistic and both resolved mutants are fitted"** (launch.md:134–138).
- **Held out:** only a description-level reading of Scheer's held-out dates (small lawn, kinematics
  and activity only), plus the tax-4 leave-one-out, which gates nothing (launch.md:150–179). The
  PD1074 held-out dates are the source of the primitives' on-food reversal values and the
  speed-following knots (launch.md:171), so only N2's 26 unread dates are clean. Chemotaxis and
  López-Cruz held-out batches are not read (launch.md:173).
- **Status:**
  - **Pilot 1 was killed** at 2a, D_fit 3.13 > 2.5 (launch.md:266; `pilot_first.json` "killed: 2a:
    mean D_fit 3.13 above 2.5").
  - **Part A was amended** 2026-11-12 (six starts; stage 1 shrunk).
  - **Pilot 2 kept no start of 6**, all dropped at gen 60, 15.8 h (launch.md:305–316;
    `pilot_second/record.json` "killed: no start of 6 held the lawn").
  - **Amended again** 2026-11-13 (the lawn guard). The bars were set by reading the earlier runs: "The
    0.95 bar and its three generations were read off the first six rows" (launch.md:284).
  - **Pilot 3** (base 1.14e9) is running. Nothing about it is committed.
  - *Off-repo and uncommitted, flagged:*
    - `build/c2-fit/pilot.log` at 2026-11-14 00:28 shows start 0 in stage 2a at generation 74,
      having gone past gen 60. By the rule, that implies it was kept; I did not read the readouts in
      `state.pkl`.
    - `build/c2-fit/guard_rep_3.log` shows that the guarded re-run of pilot 2's start 3 ended at gen
      300 at D_fit 4.23, "does not reach the bar". So under the guard, 2 of 3 unguarded-stage-1
      starts reached the bar (0 and the probe), and start 3 did not. Launch.md:325 had left that
      open.
  - F.0b-ii, the registered fit and the leave-one-out are all unrun (launch.md:262).
- **Slow-noise positive control: passes** on LB 131's answer, roaming p90 rise +4.90 at 30 s
  against ≥ 3 (launch.md:358–370).
- **Cost:** about 117 h for pilot + recovery + fit + leave-one-out (launch.md:383).

## 3. The benchmark changed after controllers' results

Each revision passed registered validity checks: held-out real worms pass, and every negative fails
5/5 (LB 115:17–27; LB 120; LB 124:3–22). Every revision since v1 was nonetheless triggered by a
controller's result:

- **v2** followed C.1's v1 pass "partly through routes no real worm takes" (LB 115:45–50).
- **v3** followed C.1b's reversal-dominated answers. "The candidate fix is in the benchmark" was the
  maintainer's decision (LB 119:193; LB 120:42).
- **v4** followed C.2's stage-2 answer that does not run (LB 124:42; LB 126:41). Its three activity
  statistics were chosen by a planning probe that scored C.2 stage 2, C.1 v3 and the hazard
  controllers (LB 124:66–78). C.1's passing v3 answer then failed v4 (LB 124:15).
- **The body model changed too**, though that is not a benchmark revision. The speed-following
  on-food reversal (LB 131/132) was measured from held-out data, but it was introduced because C.2's
  fit sat on a frontier at 8–9 of 15. It turned a 3/10 into a 15/15 pass after a refit (LB 131:163–186).

Activity-family context from LB 124:72–76, in between-date SDs from the real value:

| Statistic | C.2 stage 2 | C.1 v3 | Constant hazards | Pirouette hazards |
|---|---|---|---|---|
| pause share | +1.6 | +1.3 | −5.0 | −6.4 |
| flip fraction | −4.8 | −1.4 | −3.8 | −4.7 |
| onset rate | −4.6 | −3.1 | −2.8 | −4.1 |

B.4's six negatives, v2's blind worm (D 3.6–4.6) and C.1's v1 answer fail every version 5/5
(LB 115:25–27; LB 124:9–10).

## 4. Candid flags: researcher degrees of freedom and held-out status

01. **The held-out families were moved into the objective after they failed.**
    - *D37 as written:* C.2 holds out local search, and from v2 also chemotaxis; the mutants are
      predictions for every controller (LB 115:56; LB 113 encodings.md:3).
    - *LB 126:* both held-out families and tph-1/tax-4 fail.
    - *LB 127:* all six families go into the fit.
    - *LB 128/129:* the mutants go into the fit, after failing while held out.
    - *D37 amendment, 2026-11-10:* C.2 and C.1's refit both fit the wild type on all six families plus
      their encodable mutants.
    - **No graded family is now held out for either controller.** Launch 133 states its verdict is
      in-sample (launch.md:138). What remains is a description-level reading of held-out real dates
      (lawn, kinematics and activity only) and one tax-4 leave-one-out that gates nothing.
02. **Mechanisms were chosen by looking at the graded families.**
    - C.1: three mechanisms in LB 121 and a fourth in LB 125, "chosen by looking" (LB 121:131;
      LB 125:101).
    - C.2: the encoder options (soft cap, centres, graded box, inhibition floor 0.09, receptor signs,
      OU noise, three guards, failure-probability term, curriculum, restarts with a lawn-keep rule)
      were each added after a failed reading (LB 126–133).
    - The logbooks acknowledge this every time.
03. **Forking paths.**
    - Roughly 40+ C.2 probe fits or continuations across LB 122–133, each with a rule "fixed before the
      run", but with the sequence chosen adaptively by the maintainer after each reading (LB 131:204).
    - The first C.2 pass at 3/5 (LB 129) came only after a **hand-set CEP gain chosen from ~10 nudges
      on one population** (LB 129:131). The clean-start pass (LB 131) needed a body change plus a
      refit.
    - Launch 133's keep bar (0.95) and its generations were read off prior runs (launch.md:284).
04. **Passes sit at the threshold, with small populations.**
    - C.1 passes at exactly 3/5 in LB 113 and LB 121.
    - Selection is noisy: in LB 121, restart 4 passes 5/5 registered populations but only 2/5 selection
      populations, and restart 6 was chosen (LB 121:130).
    - Each population is 200 animals; 5 registered populations, or 15 in probes.
    - C.2's 15/15 is robust *within simulator noise* (SD ≈ 0.05–0.18 z). But it is in-sample, and the
      final stage optimises each statistic's chance of failing the bar directly (LB 131:121–129), so
      passing is close to what was optimised.
05. **Lineage.** C.2's answers in LB 126–130 descend from an imitation of C.1's unregistered v4 probe
    answer (LB 126:97; LB 129:133). Only LB 131 onward is a clean start.
06. **The perturbation tests are weak.**
    - pdfr-1 is magnitude-only (LB 128:101).
    - mgl-1 is inert and has no band (LB 127:72; LB 128:126).
    - tph-1 and tax-4 are each graded on one or two small-lawn statistics (encodings.md:9–13).
    - Every C.2 fit switches inhibition off unless floored (LB 127:81–97).
07. **What the benchmark does not grade.** The off-food reversal share is reported only. Both
    controllers under-reverse: C.1 about 3–5% and C.2 5.5–7%, against 17.3% (LB 121:132; LB 125:77;
    LB 129:56, 123).
08. **Data reuse.**
    - The López-Cruz sign overrides (LB 122:216–255) come from the paper behind the López-Cruz local
      search reference. They were added while local search was still held out for C.2; that is now
      moot.
    - PD1074's held-out dates set the primitives (LB 108, LB 132:48), which weakens launch 133's
      held-out reading on them (launch.md:171).
09. **Seed overlaps** are recorded and judged harmless, since they touched description only
    (LB 113:82–84; LB 131:208).
10. **F.0b-ii will test reach, not identifiability.** It has one truth, next to the wild type
    (launch.md:124). In LB 121's recovery, single parameters already missed by up to 0.86–0.88 of
    their range while behaviour matched (LB 121:102).

## 5. Could not verify

- The readouts of the running third pilot (in `build/c2-fit/pilot/state.pkl`, not opened). I
  inferred "start 0 kept" only from 2a continuing past gen 60.
- The off-food reversal share of LB 131's final answer.
- LB 116/117 numbers come from AGENTS.md and memory. I did not re-read those logbooks.
- I spot-checked only the JSON fields listed above; per-statistic z values were not re-derived.
