---
name: nematode-review-impl
description: Comprehensive review of implementation before evaluation or merge. Checks completeness, correctness, code quality, test coverage, and more.
metadata:
  author: nematode
  version: '1.0'
---

Comprehensive review of implementation before evaluation or merge.

**Input**: Optionally specify which files or features to review. If omitted, review all uncommitted/recent changes.

**Steps**

01. **Identify scope**

    Determine what to review:

    - If an OpenSpec change exists, use its task list to identify expected files
    - Otherwise, use `git diff --stat main` to find changed files
    - Read all new/modified source files thoroughly

02. **Completeness check**

    - Are all tasks/requirements implemented?
    - Are all registration steps done? (`dtypes`, `__init__`, `config_loader`, `brain_factory`, `BrainConfigType`)
    - Are example configs created?
    - Are smoke test entries added?
    - Are docs updated? (AGENTS.md, openspec/config.yaml, README.md if applicable)

03. **Correctness check**

    - Does the implementation match the spec/design?
    - Are algorithms implemented correctly? (PPO, GAE, BPTT, gradient clipping)
    - Are edge cases handled? (empty buffers, episode boundaries, None fields)
    - Is gradient flow correct? (detached critic, separate optimizers, proper backward passes)
    - Are there off-by-one errors in chunk boundaries or buffer indices?

04. **Code quality check**

    - No dead code or unused imports
    - No duplicated logic that should be shared
    - Consistent naming with existing codebase
    - Proper error messages for validation failures
    - Logging at appropriate levels

05. **Performance check**

    - No unnecessary tensor copies or conversions
    - No accidental gradient computation in inference (torch.no_grad)
    - Buffer sizes and data structures are appropriate
    - No memory leaks (detach/clone where needed)

06. **Test coverage check**

    - Config validation (defaults, custom values, invalid values rejected)
    - Core functionality (construction, single step, multi-step, learning)
    - Edge cases (episode boundaries, buffer overflow, weight persistence round-trip)
    - Integration (sensory module compatibility, protocol compliance)
    - Are there missing test scenarios?
    - **Planted defects:** break the code each new test, guard and parity check covers (a wrong
      sign, a dropped stage, a skipped hand-off, a guard's marker quoted in a record) and confirm the
      test fails. A test that passes on broken code is not evidence; record how many planted bugs
      the first tests missed and fix the tests until none survive.
    - Do the tests' inputs exercise the feature's range? A parity test whose inputs never vary the
      quantity under test passes vacuously.
    - Do seed blocks have a test that every pair is disjoint over its full span, and do smokes and
      probes that reach held-out code pass band-cohort stand-ins rather than the defaults?

07. **Instrument identity check** (any change to the simulation, grading or connectome code path
    that claims to leave outputs unchanged)

    - C.1's simulation path, with `scripts/analysis/speedup_identity_check.py`: on the code before
      the change, `baseline`, then `check` on that same code, so a stored record that no longer
      reproduces is not blamed on the change; then `check` again after the change. Give every run
      this review's own `--build <dir>` and `--out <json>`: the defaults are Logbook 118's
      baseline directory and its committed `identity.json`, which a review must not overwrite;
    - the connectome brain, with `scripts/analysis/connectome_identity_check.py baseline --build <dir>`
      on the code before the change and `... check --build <dir> --out <json>` after it, with the
      same `<dir>` (the script's own example is `build/c2-encoder/identity`);
    - grading code moved or edited in `packages/wormbench/`, with
      `scripts/analysis/wormbench_identity_check.py`: `baseline` in a tree at the code before the
      change, then `check` in a different tree at the changed commit (it refuses the baseline's
      commit). Run each stage from its own tree with that tree's interpreter,
      `cd <tree> && .venv/bin/python -I <script> <stage> --tree . --build <shared dir>`, with one
      shared build directory outside both trees;
    - after changing a file the package copies, `scripts/analysis/wormbench_records.py check` (and
      `wormbench_example.py check`).

08. **Security / robustness check**

    - No hardcoded paths or secrets
    - Input validation on public APIs
    - Graceful handling of unexpected inputs

09. **Run verification**

    ```bash
    uv run pytest -m "not nightly" --tb=short -q
    uv run ruff check <changed_files>
    uv run pyright <key_files>
    ```

10. **Report findings**

Categorise as:

- **Bugs**: Must fix (incorrect behaviour)
- **Issues**: Should fix (code quality, missing tests)
- **Minor**: Nice to have
- **Verified correct**: Things explicitly checked and confirmed good

For each bug/issue, propose a fix.

**Guardrails**

- Read the actual code, don't just check file existence
- Cross-reference with existing similar implementations (e.g., compare lstmppo with mlpppo patterns)
- Run the test suite — don't just review, verify
- Check pyright on modified files
- When reviewing your own branch, also run a reviewer subagent with no context (give it the code
  and the spec, not your conclusions), as `docs/research/phase-protocol.md` principle 4 asks before
  any launch
