---
name: nematode-review-spec
description: Review an OpenSpec change for completeness, correctness, and scientific rigor before implementation. Use before starting implementation of a change.
metadata:
  author: nematode
  version: '1.0'
---

Review an OpenSpec change to ensure it is ready for implementation.

**Input**: Optionally specify a change name. If omitted, infer from context or list available changes.

**Steps**

1. **Load the change**

   Read all artifacts:

   - `openspec/changes/{name}/proposal.md`
   - `openspec/changes/{name}/design.md`
   - `openspec/changes/{name}/specs/**/*.md`
   - `openspec/changes/{name}/tasks.md`

2. **Review against criteria**

   Check each criterion and report findings:

   **Scope & Completeness**

   - Does the change cover what was originally planned?
   - Are there missing deliverables or scenarios?
   - Are edge cases addressed?
   - Does the task list cover all spec requirements?

   **Logical Consistency**

   - Do the design decisions follow from the problem statement?
   - Are there contradictions between artifacts?
   - Do the tasks implement what the specs describe?
   - Is the dependency order of tasks correct?

   **Correctness**

   - Are the technical claims accurate? (e.g., PPO formulation, BPTT approach)
   - Are referenced patterns from existing code still valid? (check that referenced files/lines exist)
   - Are default values sensible?

   **Scientific Rigor** (where applicable)

   - Are hypotheses testable?
   - Are biological claims cited or justified?
   - Are performance expectations realistic given prior results?
   - Are comparison baselines identified?

   **Campaign Readiness** (any change that launches a campaign; blocking if missing for one over ~2 h)

   - Has every gate the registration reads been evaluated at the registered point **before launch**?
     Run `scripts/campaigns/gate_preflight.py --panel <analysis module> --logs <runs at that point>`
     on committed runs where they exist, otherwise on a pilot. Any `saturated`, `fails_floor`,
     `near_bar` or `no_evidence` level blocks the launch until resolved.
   - Was a gate carried over from another cell or learner re-checked on this one?
   - Is the cost estimated from a pilot configured the way the campaign will be (same episodes, same
     arm mix), not from a lighter run or another panel's average?
   - Did every pilot and smoke run use seeds disjoint from the registered band? Is every seed block
     recorded by its full span (base + stride × largest index), with a test that every pair is
     disjoint?
   - Was each gate preflighted **itself** — the stage's own function, criterion by criterion — not a
     summary such as an RMS? Was a positive control's bar met by its oracle (the generating
     parameters re-simulated with fresh noise), and is a bar between simulations set against the
     simulator's own same-parameter noise?
   - Does a smoke or probe that reaches held-out code pass band-cohort stand-ins rather than the
     defaults?

   **Fitting Readiness** (any change that fits a model to data; principles 2, 4, 7, 10 and 11 of
   `docs/research/phase-protocol.md`)

   - **Representability:** has the model been shown able to produce the target at all — by forward
     passes, output ranges or hand-set changes from its start or existing answers — before the fit,
     screen or control is designed? Does a control's answer actually express what it controls?
   - **The objective against the verdict:** is the objective broken down term by term at its start
     and on pilot candidates? Does every graded statistic have slope there (no term capped or flat,
     none cheaper to leave undefined than to miss), and does a statistic leaving its bar cost more
     than the score's noise between candidates?
   - **The recovery gate:** can it fail? Score what a non-recovery would leave — the fit's default
     target (such as the bands' centres) and a match on one term alone — against the bar. Are the
     recovery's keep and drop rules scored against the same reference as its objective?
   - **Runs behind the procedure:** if the procedure was amended across failed pilots, has it met its
     go rule on at least two fresh runs? If not, the registration must say the go rule is one draw
     **and** register a recovery control, run by the same procedure, as the second run; saying it
     is one draw without that control does not pass.
   - **Fitted or predicted:** does the registration say, for each perturbation, whether it is fitted
     or predicted, and does nothing read a fitted response as a prediction?
   - **Benchmark version:** is the benchmark version the fit and any table read fixed before the
     fit registers, and are earlier passes on older versions read as conditioned?

   **Code Quality Concerns**

   - Will the proposed implementation integrate cleanly with existing code?
   - Are there potential circular imports, type issues, or API mismatches?
   - Is the test coverage plan sufficient? Does it plan to run the tests and guards against
     deliberately planted defects, with inputs that exercise the feature's range?
   - Does a change to the instrument's code path (a speed-up or a refactor that claims to change
     nothing) plan a byte-identity check, run first on the code before the change
     (`speedup_identity_check.py`, `connectome_identity_check.py`, `wormbench_identity_check.py`)?

   **Missing Items**

   - Registration steps (`dtypes`, `__init__`, `config_loader`, `brain_factory`)?
   - Documentation updates (AGENTS.md, openspec/config.yaml)?
   - Example configurations?
   - Smoke test entries?

3. **Report findings**

   Categorise as:

   - **Blocking**: Must fix before implementation
   - **Should fix**: Improve before implementation
   - **Minor**: Can address during implementation
   - **Notes**: Observations, not issues

4. **Suggest fixes**

   For each blocking/should-fix issue, propose a specific fix.
   Ask the user if they want fixes applied.

**Guardrails**

- Check that referenced files/functions actually exist in the codebase
- Cross-reference task descriptions against spec scenarios
- Verify the task count seems reasonable for the scope
- Don't just rubber-stamp — look for real issues
