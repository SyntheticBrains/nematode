---
name: nematode-review-spec
description: Review an OpenSpec change for completeness, correctness, and scientific rigor before implementation. Use before starting implementation of a change.
metadata:
  author: nematode
  version: "1.0"
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

   **Code Quality Concerns**
   - Will the proposed implementation integrate cleanly with existing code?
   - Are there potential circular imports, type issues, or API mismatches?
   - Is the test coverage plan sufficient?

   **Missing Items**
   - Registration steps (dtypes, __init__, config_loader, brain_factory)?
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
