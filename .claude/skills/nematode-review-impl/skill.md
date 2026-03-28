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

1. **Identify scope**

   Determine what to review:

   - If an OpenSpec change exists, use its task list to identify expected files
   - Otherwise, use `git diff --stat main` to find changed files
   - Read all new/modified source files thoroughly

2. **Completeness check**

   - Are all tasks/requirements implemented?
   - Are all registration steps done? (`dtypes`, `__init__`, `config_loader`, `brain_factory`, `BrainConfigType`)
   - Are example configs created?
   - Are smoke test entries added?
   - Are docs updated? (AGENTS.md, openspec/config.yaml, README.md if applicable)

3. **Correctness check**

   - Does the implementation match the spec/design?
   - Are algorithms implemented correctly? (PPO, GAE, BPTT, gradient clipping)
   - Are edge cases handled? (empty buffers, episode boundaries, None fields)
   - Is gradient flow correct? (detached critic, separate optimizers, proper backward passes)
   - Are there off-by-one errors in chunk boundaries or buffer indices?

4. **Code quality check**

   - No dead code or unused imports
   - No duplicated logic that should be shared
   - Consistent naming with existing codebase
   - Proper error messages for validation failures
   - Logging at appropriate levels

5. **Performance check**

   - No unnecessary tensor copies or conversions
   - No accidental gradient computation in inference (torch.no_grad)
   - Buffer sizes and data structures are appropriate
   - No memory leaks (detach/clone where needed)

6. **Test coverage check**

   - Config validation (defaults, custom values, invalid values rejected)
   - Core functionality (construction, single step, multi-step, learning)
   - Edge cases (episode boundaries, buffer overflow, weight persistence round-trip)
   - Integration (sensory module compatibility, protocol compliance)
   - Are there missing test scenarios?

7. **Security / robustness check**

   - No hardcoded paths or secrets
   - Input validation on public APIs
   - Graceful handling of unexpected inputs

8. **Run verification**

   ```bash
   uv run pytest -m "not nightly" --tb=short -q
   uv run ruff check <changed_files>
   uv run pyright <key_files>
   ```

9. **Report findings**

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
