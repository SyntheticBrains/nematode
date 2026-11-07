*Committed with [Logbook 127](../../127-c2-readability.md) as written on 2026-11-06, after Logbook 126 and before the readability probe; one line on the compute decision was edited the same day. It was the maintainer's re-assessment of Phase 10's plan. Its records are now in `probes/` beside this file and in Logbook 126's, not `build/`. The maintainer chose its re-plan with the probe first, and the compute decision after it. The probe's answer is the logbook's.*

# Phase 10 re-assessment (2026-11-06)

For the maintainer, after Logbook 126 (PR #522). Nothing here is registered. Figures come from committed records, except one quick check run today, marked as such.

## Where the phase stands

- **The minimum viable result is already met.** Benchmark v4 is VALID (Logbook 124), and E + T, track import and N are shipped. The benchmark is citable on its own.
- **C.1** passes v3 (Logbook 121). On v4 it needs `r_dwell_on_food`, and its refit is deferred to C.4 (Logbook 125).
- **C.2** fits v4's D_fit in probes (D_fit 1.29–1.40, 5 of 5 populations). Every held-out family fails, and so do tph-1 and tax-4 (Logbook 126).
- **Left to do (all MUST):**
  - C.2's procedure and its readiness restarts;
  - F.0b-ii;
  - C.3;
  - C.4, with C.1's refit;
  - W: 6 arms × 8–16 draws;
  - the synthesis.

## What changes the calculation: identifiability, not search

Logbook 126 solved the search problem on the fitted families. What it exposed is that the fitted families do not determine the mechanism.

- **Today's quick check** (description: one 30-min episode on Scheer's lawn, the profile seed, the guarded answer):
  - The fit does use its modulator layer. Many strengths sit at their box bounds, and the fitted timescales are 459 s for the monoamines and 26 s for the neuropeptides.
  - But serotonin's pool is nearly constant, at 0.0134 ± 0.0036. Knocking out tph-1 removes a near-constant offset: reversals go from 17.0% to 15.7% of seconds, and runs from 82.6% to 84.3%.
- **Why this happens.** Many internal configurations reproduce the wild type's on-lawn statistics, and nothing in D_fit chooses the one the mutants test.

**What that means for W, if every arm is fitted on D_fit only:**

| W level | Likely reading | Why |
|---|---|---|
| Fitted families | "the nulls pass too" | The wild type reaches D_fit 1.29–1.40, well inside W's minimum of 2.597. A 136-parameter network will probably bring the nulls inside it too. Phases 8 and 9 found nulls that learn as well. |
| Held-out families | floor | The wild type fails every one, so no null can do worse in a way that reads. |
| Mutants | inert or `unidentified` in every arm | Nothing in the fit pins the pathway the mutant removes. |

W would then read null on every level, for a reason that concerns the fit, not the wiring. In the protocol's terms, a level the wild type cannot pass sits at the floor and is unreadable. The current plan would find this out after roughly 500–1,000 h of fitting.

## Other problems with the current plan

1. **C.2 and C.1 are fitted on different families.** C.1's ranking score covers all of v4's families (`c1_v3_fit.py` scores the bands' full family map). C.2's D_fit leaves out local search and chemotaxis. So C.4's "C.2 against C.1" would compare a model fitted on search with one that never saw it.
2. **Cost.** A fit takes about 10 h as run: the imitation 3.3 h, the refinement 1.4 h, the guarded search 5.5 h. W's 48–96 fits come to about 480–960 h, before the readiness restarts and F.0b-ii.
3. **The imitation ties C.2 to C.1,** and it adds 3.3 h to every arm.
4. **The benchmark has changed often.** It went from v1 to v4 in 10 days. Each revision was justified by its controls, but each one restarted C.1's and C.2's work.

## Recommendation: plan C.2 and W around a perturbation test the wild type can pass

01. **Fit the wild type on every family, as C.1 is** (a D37 amendment, dated, made before any C.2 fit is registered).
    - Local search's decay needs the slow pools to vary, and the off-food families constrain the off-food behaviour that is now wrong.
    - It makes C.4's contrast symmetric.
    - **Cost:** 2.6× per generation (the cost pilot: 55.6 s against 21.2 s at population 16), or about 12–15 h a fit at population 48.
02. **Hold out the perturbations, which is where wiring should show.** These are Scheer's three mutants, plus **López-Cruz's mgl-1**.
    - **Its data:** 11 rows across many dates, with a strong phenotype. Early reorientations are about 0.9–1.1 per 2-min bin, against 2.6–2.8 for N2 recorded on the same dates. There are enough dates for a band built across dates and a positive control on a held-out date.
    - **The AIA and ADE silencing and rescue rows** have one date each, so they are reported beside the grade, not graded.
    - **Encoding:** mgl-1 can be encoded in C.2, through the MGL-1-mediated glutamate edges onto AIA and ADE that the 16 sign overrides already mark. It cannot be encoded in C.1, so it is a prediction only the connectome can make.
    - **What it tests about wiring:** in a null those edges move with their counts, so knocking out mgl-1 hits different cells.
    - **Caveat:** the overrides' signs came from the same paper's mechanism. So mgl-1 tests what the fitted network does with them, not the signs themselves.
03. **Check readability before anything registers.** Run one warm probe on discarded seeds: from the guarded answer, about 150 generations on all six families (about 8–9 h). It reads two things:
    - **(a) Can C.2 fit all six families?** If not, the claim scopes to the lawn assays (the roadmap's "slow variable" risk row).
    - **(b) Do tph-1, pdfr-1, tax-4 and mgl-1 move, and in which direction?** If they stay inert even then, escalate: fit one perturbation (for example tph-1) and hold out the rest.
04. **Cut W to four core arms:** the wild type, degree-preserving, boundary-held and receptor-placement nulls (a D40 amendment).
    - Gap-only and the class-preserving left–right null are named as not run; the latter is not built.
    - **Cost:** 4 arms × 6 draws × 12–15 h, or about 290–360 h. A sequential stop on the draws can end it early.
05. **Decide the compute before W registers; the probe sizes it.** On this machine W alone is about 2–3 weeks of continuous running. Rented CPU running fits in parallel would take days.
    - **Caveat:** byte-identity has only been checked on this machine. Either run every W arm on one platform, or check identity across platforms first.
06. **Freeze benchmark v4** for the phase's graded wild-type statistics.
    - mgl-1 enters as a held-out perturbation family with its own controls. It was a planned addition ("López-Cruz perturbation family before W"), not a revision.
    - Any later loophole is reported beside the grade, unless a registered negative passes.
07. **Drop the imitation for registered fits** if the readiness restarts converge without it. The ablation escaped the basin unaided.
08. **Keep the machine busy during engineering.** Run C.1's v4 refit (12–18 h) and the probe while C.2's procedure is being built.
09. **Reduce F.0b-ii** to 2 truths × 2 restarts. Make one truth the wild type's answer and one a null's: a null truth checks that the method can fit rewired networks, which is what a reading of "a null fits worse" relies on.
10. **Size C.3 from a pilot.** Use 8 seeds if 16 is too slow; that needs a D39 amendment only then.

## Odds (judgment, not measurement)

- **The current plan:**
  - W readable on any held-out level: about 10–20%.
  - W read at all (Target, in its weak form): about 50–60%, mostly a compute and schedule risk.
- **The re-plan:**
  - C.2 fits all six families: about 50–65%.
  - Given that, the wild type phenocopies at least one held-out perturbation: about 35–50%.
  - So a readable W: about 20–35%. The probe tells which way it goes within a day, not after W has run.
- **Either way,** the minimum viable result stands.

## Alternatives considered

- **The current plan, unchanged.** Its cost and readability are as above.
- **Shrink W to a pilot** (the wild type and one null, 3 draws each), make the fidelity table the headline, and defer the full W to Phase 11.
  - It finishes fastest, in about 2 weeks, but says least about the connectome.
  - It meets Target only as "W read with its sensitivity limit stated".
- **Fit a mutant from the start** (system identification). It is the escalation in step 3(b), not the first step, because it spends a held-out test.
- **Gradient-based fitting through a differentiable simulator.** Weeks of engineering, for a search problem that is already solved.

## Decisions for the maintainer

1. **The direction:** the re-plan, the current plan, or the shrunk W.
2. **The fit split:** all wild-type families (with or without chemotaxis), with the perturbations held out.
3. **mgl-1 added** to the held-out set.
4. **W's arms:** 4 core arms, against 6.
5. **Compute:** local or rented.
6. **Freeze benchmark v4.**
