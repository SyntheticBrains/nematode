# How a phase is run

*Recorded 2026-09-12, after Phase 7a's block I. This is not a list of rules invented after a
failure. It is the pattern Phases 5 and 6 used, the one place Phase 7 departed from it, and why —
kept so that the departure is recognisable the next time it is about to happen.*

*(Added 2026-10-25.) The principles were written around learning rules, but Phase 10 fits
controllers to real-worm data and grades them against it, and the principles carry over. The
additions dated 2026-10-25 come from that work: a benchmark built from data, its controls, and the
fitting methods' positive controls.*

## The pattern that worked

Phase 6 ran a positive control ([Logbook 030](../experiments/logbooks/030-bit-memory-positive-control.md))
**before** the memory-axis programme it gated. Phases 5 and 6 built a ladder from oracle sensing to
biologically honest sensing and measured the gap at each rung. Phase 6 wrote a go/no-go decision at
every gate. Each of those is one of the principles below; none of them was new when Phase 7 began.

## Where Phase 7 departed, and why

Phase 7's positive control is [Logbook 048](../experiments/logbooks/048-l4-rule-positive-control.md)
— eight logbooks after its first panel, 040. Seven registered panels ran before anyone checked that
the rule could learn anything, and the control then showed it could not: its updates were nearly
orthogonal to the policy gradient.

The departure had a specific cause, and it is the failure mode most likely to recur. L4 put a **new
rule** on an **already-validated substrate**. The connectome had cleared Phase 6's gates under PPO,
so the platform felt tested, and that confidence carried over to the rule, which had never been
shown to learn anything. The positive control was skipped not from ignorance of the practice but
because the thing that needed it looked as though it had already passed.

> **A new component on a validated platform still needs its own positive control. The platform's
> validation does not transfer to it.**

This is the same trap as Phase 6's grid-versus-continuous non-commensurability lesson, one level
down.

## The principles, each with the cost of ignoring it

### Before building anything

1. **Name the question and both answers.** Register what a positive and a negative result look like
   and what each licenses, before any data exists. *Phase 7 did this consistently, and it is why
   the negatives are citable rather than merely disappointing.* *(Added 2026-09-20.)* **Name every
   branch, not only the two that matter.** An outcome map that omits the combination nobody expects
   invites it to be read as whichever neighbour is convenient; the branch where two contrasts point
   opposite ways licenses no follow-on work and needs its own registration to act on
   ([Logbook 042](../experiments/logbooks/042-l4-panel3.md)). And where a diagnostic borrows a
   component from the method the deliverable rules out — a tensor from a gradient-trained run — the
   record states **before it runs** that it cannot satisfy that deliverable whatever it returns, and
   names what a positive would license instead
   ([Logbook 062](../experiments/logbooks/062-l4-frozen-readout.md)).

2. **Do the feasibility arithmetic.** Ask whether theory says the method can work at this scale and
   horizon, and write the estimate down. *Node perturbation's learning speed scales as roughly 1/N
   in the number of perturbed units (Werfel, Xie & Seung 2005). At 302 units, four draws per step
   and 2400-step episodes the substrate was near a worst case; nothing in the design considered it.
   The estimate would have cost an hour. Measuring it cost a day and confirmed it.* *(Added
   2026-09-19.) For a correlation, the arithmetic compares the predictor's spread across the units
   the test runs over with the contrast that motivated the hypothesis. L.4/L.5's structural probe
   was motivated by the wild type sitting at 0.154 against rewirings at 0.01–0.04, and registered
   across 96 rewirings whose whole spread is 0.014–0.027 — a valid test of a narrower question
   than the one it was meant to license, [Logbook 067](../experiments/logbooks/067-l4-feature-ablations.md).*

3. **Establish that the task is solvable with the strongest available method.** Know what "learned"
   looks like before asking a constrained method to produce it. *The phase learned that PPO solves
   this task from random weights — 68.5% on the wild type, 81.2% on the rewired null — at its tenth
   item (Logbook 043), after seven panels had asked a local rule to do it. That number also bounds
   the rule: from random weights it reaches 17.8% and sits below its own unmodulated floor. An
   earlier reading of the same table, that PPO also destroys competent policies, was withdrawn in
   Logbook 056 — warm-starting hurts PPO, which is a fact about warm-starting, not about the task.*

### Instruments before substrates

4. **Positive control first.** The method learns a minimal task with closed-form bounds before it
   touches the real substrate. *Ran twelfth. Cost seven panels.* The same discipline applies to a
   **metric**: a statistic that is meant to exclude a mechanism has to be run first on a known
   instance of that mechanism. *External precedent, 2026-09-13: a fly connectome analysis
   (`pwang724/fly-circuit-exploration`) excluded activity-based memory with a recurrence metric,
   then found the metric gave the same answer on the EPG ring attractor — a measured
   persistent-activity network — and withdrew the exclusion in its own audit. The public claim
   went out the next day with the exclusion still in it.* *(Added 2026-09-20.)* Two corollaries.
   **A substrate result is read against the control**: where the control has not run or has not
   passed, a null is consistent both with the substrate carrying no signal and with the method being
   unable to show one, and the record says so rather than picking
   ([Logbook 056](../experiments/logbooks/056-l4-ladder-reread.md)). **A control with a closed-form
   optimum names the arm that must fail as well as the one that must pass** — either expectation
   being violated voids the control rather than producing a result
   ([Logbook 048](../experiments/logbooks/048-l4-rule-positive-control.md)).
   *(Added 2026-10-25.)* Three more corollaries.

   - **A positive control's success bar is met by its oracle before launch.** Score the known
     answer against the bar: the generating parameters re-simulated with fresh noise, or the exact
     solution. If the oracle cannot pass, the control tests nothing, whatever the method does.
     *F.0a's informative gate asked simulations at NPE's posterior draws to reach a median D of at
     most 1.42 from the truths' statistics. Two simulations of the same parameters already differed by a median
     D of 4.08, so no posterior could pass. A five-hour run was spent on a check two minutes of
     re-simulation would have shown was out of reach. The gate had been added in the change's own
     review, and the launch record noted that the preflight tool "does not apply", which was true
     of the tool and not of the principle,
     [Logbook 111](../experiments/logbooks/111-npe-recovery.md).*
   - **An instrument is shown to fail what it must fail, part by part.** A graded family or
     statistic that a trivial negative passes cannot carry a verdict alone, even when the whole
     instrument rejects that negative. Check each part against a negative built to lack the thing
     it measures, before registering what rests on it. *A population with no local search at all
     passed the local-search family 12–13% of the time, which the benchmark's hardening found
     before the controls ran and narrowed with a registered presence test
     ([Logbook 109](../experiments/logbooks/109-benchmark-grading.md), addendum). The untrained
     connectome passed both the family and the presence test (bound 1.006), so the distance used for
     fitting leaves local search out ([Logbook 110](../experiments/logbooks/110-benchmark-controls.md)).*
   - **Where validity needs several controls to pass, register the joint rule.** Fifteen control
     units each passing at its own 95% would void an instrument by chance about half the time.
     Register an allowance from the units' failure rates, measured where possible, before any
     held-out value is read, and report which units failed. *B.4's allowance of two failures
     carried a 2.5% chance of a spurious void, against 49% for requiring every unit
     ([Logbook 110](../experiments/logbooks/110-benchmark-controls.md); the 49% is in the change's
     [proposal](../../openspec/changes/archive/2026-10-25-add-benchmark-controls/proposal.md)).*

5. **A task ladder from the control to the real task**, the method cleared at each rung before the
   next. Easy to hard: foraging, then foraging with predators, then with thermotaxis. Never jump
   orders of magnitude. *All 31 plastic configs sit on the hardest task. The only intermediate rung
   was the one-step control, and its delayed extension stops at twenty steps against episodes of
   244 to 2400. The gap between them is where every candidate explanation went to die.*

6. **Controls presuppose an effect.** Establish that the arm beats its own floors before running a
   contrast against a null. *The degree-preserving rewired null was well built on a premise nobody
   tested; panel 3 found the null's Hebbian floor sitting above the plastic arm.*
   *(Added 2026-09-19.) The same principle applies when a calibration moves the operating point:
   re-establish the effect at the new point before ablating it, and never inside the same campaign.
   L.4's registered rate check moved the atlas arms to 0.0001, and the ablation ran against a
   rate-matched baseline nobody had measured, bundled into the same 960 runs. The wide arms' own
   pilot at 0.0001 (seeds 101–104) had already shown the null's gain over floor nearly doubling,
   +7.8 to +15.0 foods, against the wild type's +13.7 to +17.3, and the launch record deferred
   reading it to the campaign; 192 runs would have read the baseline first and shown there was no
   wild-type effect at 0.0001 for the signs to carry,
   [Logbook 067](../experiments/logbooks/067-l4-feature-ablations.md).*
   *(Added 2026-09-20.)* A **replacement** needs a control of the same magnitude and arbitrary
   direction, or "this particular substitute helps" cannot be separated from "the default was bad and
   anything would help". *At matched norm the readout substitutions ran random 9.639, anatomical
   8.265, PPO's own direction 6.123 — the scale did the work and the direction was worst where it was
   expected to be best, [Logbook 062](../experiments/logbooks/062-l4-frozen-readout.md).*
   *(Added 2026-09-21.)* **And an arm must be shown to change only what it names — at the point the
   arms diverge, not only at construction.** A manipulation that shares mutable state with anything
   else carries a second difference downstream, and the initial parameters do not show it: they are
   identical, and the runs are not. *A.1's two shared-initialisation modes were built to differ from
   the baseline in one thing, which edge each drawn value lands on. The `dense_mask` mode drew a
   dense matrix, 91,204 values against the baseline's 3,709, from a generator the rollout buffer
   also consumes for its minibatch permutation — so it moved PPO's minibatch order as well as the
   weights. Two manipulations under one name. The suite asserted bitwise identity of every parameter
   the mode did not claim to touch, and passed, because the divergence only exists once training
   starts. It was found in review, after a 768-run panel, and cost a 256-run re-run of the affected
   arms; the other sixteen arms were provably unaffected and were kept,
   [Logbook 070](../experiments/logbooks/070-init-sharing-control.md).*
   **The check that would have caught it**: name the state the arms share, and assert the arms leave
   it in the same condition — the next value a shared generator will yield, the same optimiser step
   count, the same buffer contents. Where an arm consumes a shared resource by a different amount,
   it is not a one-variable manipulation however carefully its parameters were matched.
   *(Added 2026-10-05.)* **Every gate a registration reads is evaluated at the registered point
   before launch** — floors, the saturation bar, anything that can make a level unreadable — on
   committed runs at that point where they exist, otherwise on a pilot (`gate_preflight.py`). A gate
   carried from another cell is re-checked on the new one. *A.6t carried hard350's 90% saturation
   bar to the thermal cell and ran 1,024 runs, about 16 hours, into an unreadable panel: every level
   plateaued at 94–96%. A.1's runs at the same point, already on disk, sat at 91–92%; one minute
   reading them would have shown it, [Logbook 077](../experiments/logbooks/077-thermal-null-strength.md).*
   *(Added 2026-10-25.)* The preflight covers **every** gate, including one added during the
   change's own review. Where `gate_preflight.py` does not fit the design, the principle still
   applies: evaluate the bar by hand at the registered point, and for a positive control, score its
   oracle (principle 4). A bar on a statistic compared between two simulations is set against the
   simulator's own same-parameter noise, measured first
   ([Logbook 111](../experiments/logbooks/111-npe-recovery.md)).

### Parameters

7. **Sweep before pin.** A setting pinned on two pilot seeds is a hypothesis, not a recipe; every
   pinned value gets a registered sensitivity check on the cheapest platform where the method works.
   *`trace_decay 0.9` sat under every result for two months. No panel config set it.* *(Added
   2026-09-19.) A setting pinned at one width is a hypothesis again at another. `plasticity_rate`
   0.001 was pinned by R.2 on an 8-parameter readout and inherited by L.1's 78-parameter one without
   a sweep; L.4's rate check found the per-neuron wiring effect reverses one decade below it, which
   conditions L.1's positive on its rate, [Logbook 067](../experiments/logbooks/067-l4-feature-ablations.md).
   **L.1b then measured what the inherited pin cost**: at the calibrated rate the registered primary's
   sign **reverses** (interaction −0.0657 against +0.2818, three-way +0.3475 at q = 0.000 on 81 of 96
   seeds), and the effect the pin hid is larger than the effect it was pinned for — a width main effect
   of +0.6178 on 96 of 96 seeds where the original panel detected none. An inherited setting does not
   merely cost performance; it can set the sign of the result,
   [Logbook 068](../experiments/logbooks/068-l1b-rate-calibration.md).*
   *(Added 2026-09-20.)* And the sweep runs **where the method is known to learn**, with the pinned
   value as the baseline arm. A platform on which the method does not learn cannot show what a setting
   costs, because every arm reads alike at the floor — which is why an instrument block's own knob
   sweep had to move platforms before it could measure anything,
   [Logbook 056](../experiments/logbooks/056-l4-ladder-reread.md).

8. **Pilots on disjoint seeds.** Registered seeds stay untouched until the protocol is fixed. *Held
   throughout Phase 7, and it is why the pilots could inform registrations without contaminating
   them.* A smoke run that only checks configs execute is not a pilot: it cannot show a gate.
   *(Added 2026-10-25.)* Two extensions.

   - **Held-out data are read once, under a guard in the code.** The reading script refuses to run
     until its launch record is committed and unchanged. Anything the reading depends on, such as
     matched parameters, is bound to that commit, so it cannot drift between registration and
     reading ([Logbook 110](../experiments/logbooks/110-benchmark-controls.md)).
   - **Every seed drawn enters the phase's seed table:** pilots, diagnostics and derived seed
     schemes as well as registered bands. A seed that is used but not recorded is one a later band
     can silently overlap. *Logbook 104's calibration drew twelve blocks of 32 seeds across
     6001–7132, three of them inside W's reserved band, and the band was moved when the table was
     audited.*

### Running

09. **Cheapest platform first; the expensive one only when the cheap one moves.** Estimate cost from
    a pilot configured the way the campaign will be, not a lighter one. *The yardstick-before-
    connectome rule worked every time it was applied and saved ten-hour runs twice. The one cost
    estimate scaled from a lighter pilot was off by a factor of ten.*
    *(Added 2026-10-25.)* **And long enough to reach the machine's sustained rate.** A pilot of a
    minute or two measures a laptop before it throttles. *F.0a's 80-second pilot timed a
    simulation at 20 s and projected 4.05 h; the run sustained 29–31 s and took 4.9 h
    ([Logbook 111](../experiments/logbooks/111-npe-recovery.md)). Run the pilot for several
    minutes at full load, or scale its estimate by a measured sustained-load factor.*

10. **Register a minimum effect beside significance**, name the outcome that means stop, and
    **match the statistic and the metric to the outcome's shape**. *A
    paired rank test at eight seeds fires on the consistency of the sign, whether the shift is
    0.05 or 2.0 of ten. On a floor-adjacent platform that is a route to reporting nothing as
    something.* *(Added 2026-09-19.) **Register it for both directions of a two-sided reading.** L.1b
    required half of the original effect to credit the positive direction and set no minimum on the
    reverse one, so the reverse reading fired at 47% of that minimum — a size the guarded direction
    would have refused. The asymmetry was named in the record rather than exploited, and the symmetric
    reading stated beside it,
    [Logbook 068](../experiments/logbooks/068-l1b-rate-calibration.md).*
    *(Added 2026-09-20.)* **The shape matters as much as the size.** Where an outcome is bimodal — a
    seed reaching a competent policy or a dead one — a test of the level reports nothing when only the
    frequency moves, and a test of the frequency reports nothing when only the level does, so the
    registered contrast reads **both** components. A **graded** measure of progress is read beside the
    full-clear rate so that learning short of a clear is visible; and where the primary metric sits at
    its floor for both arms, the comparison is made on the graded reading rather than reported as no
    difference, because a metric that is zero for both cannot separate them. *Panels 2 and 3 were
    bimodal and read with neither component, leaving them unpowered at n = 16 and n = 48
    ([Logbook 041](../experiments/logbooks/041-l4-panel2.md),
    [Logbook 042](../experiments/logbooks/042-l4-panel3.md)); a later family with one member per
    component re-read seventeen contrasts and promoted none, so the committed negatives survived a
    statistic matched to the shape, [Logbook 056](../experiments/logbooks/056-l4-ladder-reread.md).*
    *(Added 2026-10-25.)* Three extensions from grading against real data.

    - **Where the claim is fidelity to data, anchor the minimum to the data.** The smallest effect
      worth registering is the distance two real cohorts show by chance. A difference smaller than
      real batch-to-batch variation has no behavioural meaning
      ([Logbook 110](../experiments/logbooks/110-benchmark-controls.md): W's minimum, 1.42 on D).
      That anchor is for a simulation graded against real data. A bar between two simulations is
      also bounded below by the simulator's own same-parameter noise, measured first (principle 6):
      *F.0a set its bar at W's minimum, and its floor was a median D of 4.08
      ([Logbook 111](../experiments/logbooks/111-npe-recovery.md)).*
    - **Match the uncertainty to the question, and check it before grading with it.** How far a new
      population may fall from a reference (a prediction) is not how well the reference itself is
      known (an estimate). Either has to be validated on held-back reference data before it grades
      anything. *The benchmark's first bands were too narrow: graded against bands from the other
      half of the dates, real populations passed 0.80 of the time against a gate of 0.90. Mutant
      directions were first read from the population scale, about twice the reference's own
      uncertainty: a prediction's scale used for an estimate's question
      ([Logbook 109](../experiments/logbooks/109-benchmark-grading.md), the first attempt and the
      addendum).*
    - **A failed registered gate is retried only under an amendment committed before the retry.**
      The amendment names exactly what changes. A bar that changes is reset from what the
      instrument can do, before any data it grades are read, never from those data, and it must
      still serve what rests on it. Held-out data already read are never read again (principle 8);
      band or calibration data the first attempt used are re-split or replaced with fresh seeds.
      The first attempt and its bar stay in the record beside the retry. *The reorientation detector's preflight failed, and its bars were
      reset from the instrument before any held-out cohort was read. The bands' rule check failed,
      and its scale was corrected with the gates unchanged. Both retries were committed before they
      ran. The bands' retry passed. The detector's passed every off-food gate and failed on food, so
      it was scoped to off-food tracks, and its record keeps the first bar of 0.70 beside the new
      ones
      ([Logbook 107](../experiments/logbooks/107-benchmark-readouts.md),
      [Logbook 109](../experiments/logbooks/109-benchmark-grading.md)).*

11. **A mechanism must predict, not describe.** A proposed explanation earns its place by a test
    that could have refuted it. *The horizon was a description of the one-step-success /
    multi-step-failure pattern until I.3b tested it on a multi-step task, where it did not
    transfer.* *(Added 2026-09-20.)* Three forms this takes. A setting found limiting **on a control**
    is tested on a task of the kind whose failure motivated it before a synthesis cites it as the
    explanation. A mechanism predicting that performance depends on a platform dimension has that
    dimension **varied where the method demonstrably learns**, before a failure elsewhere is
    attributed to it or cleared of it — *the rule solved a foraging cell at eight perturbed units and
    collapsed at 128, so the dimension was real; the connectome sits an order beyond the failing end,
    [Logbook 060](../experiments/logbooks/060-l4-perturbation-scale.md)*. And a structural statistic
    proposed to explain an unexplained result is **registered — the statistic, the predicted
    direction, and a minimum effect — before its correlation with the outcome is computed**, with a
    positive licensing a hypothesis rather than settling one
    ([Logbook 067](../experiments/logbooks/067-l4-feature-ablations.md)).

### Closing

12. **Re-read before shipping.** State which results are about the question and which are about the
    instrument, in one record, before the shipment decision. *Phase 7's I.4.* *(Extended 2026-09-19 at
    the Phase 7 close.)* **And when the phase closes, assign every criterion one status from a fixed
    vocabulary** — met, unmet-with-reason, deferred-with-destination, superseded-by-result,
    unreachable-with-reason — leaving none unmarked. **Deferred and superseded are not
    interchangeable**: a deferred criterion is still a question with a destination, a superseded one is
    not a question any more, and *unreachable* is the honest word where nothing in the phase could have
    made the criterion attemptable. The softer word for the harder case is how a close reads as tidier
    than it is. *Phase 7's own synthesis drafted the learnable-gap-junction criterion as superseded;
    review found one of its two recorded destinations untouched, and the status became deferred,
    [Logbook 069](../experiments/logbooks/069-phase7-synthesis.md).*
    *(Added 2026-09-20.)* **What the re-read may and may not do.** It classifies each result by what
    would have had to be true for its null to be informative, and does not assume that "about the
    question" and "about the instrument" exhaust the possibilities — *of 32 registered contrasts, ten
    were instrument findings, sixteen were substrate findings the instrument block does not reach, and
    five were about neither, because no optimiser tested had ever found the effect their premise
    assumed, [Logbook 056](../experiments/logbooks/056-l4-ladder-reread.md)*. Every committed verdict
    is carried unchanged beside its re-read, in the units and under the rule it was registered with; a
    re-read never converts a negative into a positive, and where it corrects an earlier reading it says
    so in the same place. **The shipment decision then names the branch it takes and why the others
    were unavailable** — a branch unreachable because a gate never opened is recorded as unreachable
    rather than unmet, and a gate whose literal condition and stated rationale come apart is recorded
    with both rather than resolved by interpretation,
    [Logbook 059](../experiments/logbooks/059-7a-shipment.md).
    *(Added 2026-10-25.)* **Every number in a record is read from a committed record or script, not
    recalled.** A diagnosis worked out in a scratch session is committed as a script before the
    logbook cites it, and the logbook's numbers are re-read from that script's output. *Logbook 111's
    diagnosis of its unreachable gate is the committed `npe_recovery.py diagnose` stage, and every
    number the logbook gives for it is read from that stage's `diagnosis.json`.*

13. **Re-aim what watches the field.** *(Added 2026-09-20.)* A close changes which questions are
    open, so anything pointed at the old ones is now pointed at nothing. The standing case is the
    literature watch: [`context.md`](literature-watch/context.md) is the entire definition of
    relevance for its scoring pass, and [`seeds.toml`](literature-watch/seeds.toml) decides whose
    citations are worth following. Both are hand-maintained, and the close is when they are re-read
    against the phase that is opening. *Neither can report that it has gone stale. A brief naming
    questions that closed two phases ago still returns a confident, well-formed digest — about the
    wrong things — and the failure is invisible precisely because the output looks exactly as it
    did when it was right.*

## What this is not

It is not a promise that following it produces a positive result. It produces a **decisive** one:
a question the evidence supports asking, answered either way in a form that can be cited. Phase 7's
path could not have produced a positive answer at all, because the thing being varied was never the
binding constraint. The principles above are what would have found that out in the first fortnight.
