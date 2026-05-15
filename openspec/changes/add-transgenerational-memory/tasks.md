# Tasks — Add Transgenerational Memory

Tasks are grouped by commit (per the plan's 8-commit grouping). Each group represents one PR-sized landing. Tests land alongside the production code they cover.

## 1. Scaffold strategy + Literal extensions (Commit 1)

- [ ] 1.1 Create `quantumnematode/evolution/transgenerational_inheritance.py` with `TransgenerationalInheritance` skeleton implementing the `InheritanceStrategy` Protocol. Top-1 elite `select_parents`, single-elite `assign_parent`, `.tei.pt` `checkpoint_path`, `kind() -> "transgenerational"`. All methods runnable but the substrate-loading paths return placeholder no-ops pending commit 4.
- [ ] 1.2 Extend `quantumnematode/evolution/inheritance.py` Protocol docstring + `kind()` Literal to include `"transgenerational"`. Update module docstring to document the fourth strategy.
- [ ] 1.3 Extend `quantumnematode/utils/config_loader.py`: add `"transgenerational"` to the `EvolutionConfig.inheritance` Literal. No `TransgenerationalConfig` block yet (that lands in commit 6 alongside the YAML).
- [ ] 1.4 Tests: `tests/.../evolution/test_transgenerational_inheritance.py` covers Protocol conformance: `select_parents` lex-tie (1 case), `assign_parent` single-broadcast (1 case), `checkpoint_path` format `gen-NNN/genome-<gid>.tei.pt` (1 case), `kind()` literal returns `"transgenerational"` (1 case). ~4 cases initially; expanded to ~14 in commit 4 as substrate-load/save paths fill in.
- [ ] 1.5 Carry forward the uncommitted `add-transgenerational-memory` rename in `openspec/changes/phase5-tracking/tasks.md` (line 223) into commit 1's staged changes.
- [ ] 1.6 Run `uv run pytest -m "not smoke and not nightly"` clean. Run `uv run pre-commit run -a` clean.

## 2. TransgenerationalMemory dataclass (Commit 2)

- [ ] 2.1 Create `quantumnematode/agent/transgenerational_memory.py` with the `TransgenerationalMemory` dataclass: `logit_bias: torch.Tensor`, `lineage_depth: int`, `source_genome_id: str`. `frozen=True`. `__post_init__` validates shape (`ndim == 1`), dtype (`float32`), and clamps `|x| ≤ 2.0`.
- [ ] 2.2 Implement `apply_to_logits(logits)` returning `logits + self.logit_bias` (broadcast over leading dims), preserving shape/dtype/device, no in-place mutation.
- [ ] 2.3 Implement `inherit_from(parents, decay_factor)` class method (or module factory). Top-1 elite semantics. Validates `0.0 ≤ decay_factor ≤ 1.0` and non-empty parents. Increments `lineage_depth`. Inherits `source_genome_id`.
- [ ] 2.4 Implement `extract_from_brain(brain, env, probe_positions, rng_seed)` telemetry-pass extraction. Deterministic on `rng_seed`. Runs on disjoint episode rollouts from those that produced fitness scores.
- [ ] 2.5 Implement serialise/deserialise via `torch.save`/`torch.load` over `.tei.pt`. Round-trip preserves all fields byte-equivalently.
- [ ] 2.6 Tests: `tests/.../agent/test_transgenerational_memory.py`. ~12 cases covering construction-time clamp, shape/dtype validation, `apply_to_logits` shape preservation + non-mutation, `inherit_from` geometric decay across F0→F3, decay-factor range validator, empty-parents rejection, serialise round-trip, missing-file load error, determinism of telemetry pass.
- [ ] 2.7 Run pytest + pre-commit clean.

## 3. LSTMPPO TEI prior application (Commit 3)

- [ ] 3.1 In `quantumnematode/brain/arch/lstmppo.py`: add `self.tei_prior: torch.Tensor | None = None` to `__init__`. Inside `run_brain()` (sampling forward pass at line 601), read `self.tei_prior` and add to actor logits before softmax (line 602) at every step. **ALSO inside `learn()` (training-time forward pass at line 747), apply the same `self.tei_prior` to logits before its softmax (line 748)** — this keeps the PPO sampling-distribution and update-distribution consistent so the ratio `exp(new_log_probs - old_log_probs)` remains well-defined. Add a defensive assertion at `learn()` entry that `self.tei_prior` shape/dtype is stable from the rollout window (raise `RuntimeError` on mismatch). Preserve byte-equivalence when `tei_prior is None`.
- [ ] 3.2 In `quantumnematode/agent/runners.py:654` (`StandardEpisodeRunner.run`): set `agent.brain.tei_prior` from the agent's lineage state (the assigned substrate's `logit_bias`) immediately before calling `agent.brain.prepare_episode()`. Set to `None` defensively when no substrate is configured. Skip non-LSTMPPO brains (the attribute is LSTMPPO-specific).
- [ ] 3.2b In `quantumnematode/agent/multi_agent.py:588` (`MultiAgentSimulation.run_episode`): apply the same set/clear immediately before each agent's `prepare_episode()` call. Mirror the single-agent rule (skip non-LSTMPPO brains; explicit `None` clear when no substrate). The single-agent and multi-agent paths must be observationally symmetric for the same substrate config.
- [ ] 3.3 Tests: `tests/.../brain/arch/test_lstmppo_transgenerational_prior.py`. ~8 cases: (a) default `tei_prior=None` byte-equivalence to pre-TEI baseline at same seed; (b) `bias=[+2, 0, 0, 0]` (4-action) elevates action-0 probability across 100 rollout steps relative to no-prior baseline; (c) setting `tei_prior=None` restores baseline; (d) `prepare_episode` does NOT clear `tei_prior`; (e) attribute survives across episode boundaries within a generation; (f) **PPO ratio ≈ 1.0 on first update of freshly-rolled-out data when `tei_prior` is set** (sampling/update distribution consistency); (g) **`learn()` raises `RuntimeError` on mid-window `tei_prior` mutation**; (h) `tei_prior = None` learn path is byte-equivalent to pre-TEI baseline gradients.
- [ ] 3.3b Tests: extend a multi-agent integration test (or add a new one under `tests/.../agent/test_multi_agent.py`) covering the symmetry — single-agent and multi-agent paths apply the prior identically at every step with the same substrate.
- [ ] 3.4 Run pytest + pre-commit clean.

## 4. TransgenerationalInheritance.inherit_from + checkpoint round-trip (Commit 4)

- [ ] 4.1 Fill in the strategy's substrate-save path. `EvolutionLoop`'s post-fitness hook for F0 SHALL invoke `TransgenerationalMemory.extract_from_brain(...)` on the F0 elite and write the substrate to `inheritance/gen-000/genome-{elite_id}.tei.pt`. **No substrate file SHALL be written for F1/F2/F3 generations** — F1+ substrates are mechanically derived from the F0 source at runtime (see 4.2).
- [ ] 4.2 `EvolutionLoop`'s pre-eval hook for F1+ SHALL load the F0 elite substrate from `gen-000/` and apply `inherit_from` `N` times where `N = current_generation` (F1: 1 decay step, F2: 2 steps, F3: 3 steps). This produces a depth-N substrate that the runner SHALL pick up via the runner code from commit 3. The mechanical-derivation design (vs. per-gen chain-storage) avoids the ambiguity of "per-gen elite substrate" since every member of F1+ shares the same depth-N substrate.
- [ ] 4.3 Extend `EvolutionLoop._inheritance_active()` (or add `_substrate_inheritance_active()`) so substrate-IO is gated on `kind() == "transgenerational"`. Keep weight-IO gating on `kind() == "weights"` byte-equivalent for Lamarckian.
- [ ] 4.4 GC retains only the `gen-000/` F0 elite substrate file. No GC-pass action needed on F1+ directories (none are created).
- [ ] 4.5 Tests in `tests/.../evolution/test_transgenerational_inheritance.py`: add ~8 more cases. Inheritance through 4 generations produces geometric decay; load-missing-file fallback; checkpoint round-trip via the strategy's path-builder; resume validator rejects switching to/from `"transgenerational"` mid-run; `kind()` gates the loop's `_inheritance_active` branch.
- [ ] 4.6 Run pytest + pre-commit clean.

## 5. Per-generation `lawn_schedule` consumer in EvolutionLoop (Commit 5)

- [ ] 5.1 Add `TransgenerationalConfig` and `LawnScheduleEntry` Pydantic models to `quantumnematode/utils/config_loader.py`. Validators: `decay_factor ∈ [0, 1]`; schedule entries cover `[0, generations)` exactly once each; `generation` indices non-negative; `ppo_train_episodes ≥ 0`.
- [ ] 5.2 Add the inheritance ↔ transgenerational pairing validator: `transgenerational.enabled=true ⇒ inheritance=transgenerational`; `enabled=false ⇒ inheritance=none`.
- [ ] 5.3 In `quantumnematode/evolution/loop.py`: just before `optimizer.ask()` at the top of each generation, if `cfg.transgenerational is not None`, look up the schedule entry for the current generation, rebuild the env config with `pathogen_lawns_enabled` set from the schedule, and override the fitness invocation's `learn_episodes_per_eval` with `ppo_train_episodes`. Gate behind `if cfg.transgenerational is not None:` so the no-op path is byte-equivalent.
- [ ] 5.4 Tests: `tests/.../evolution/test_loop_transgenerational_smoke.py`. ~6 cases (smoke-marked): 2-gen paired TEI-on/TEI-off run with pop 4 × 2 ep diverges in F1 choice index between arms; schedule with pathogen-on F0 + pathogen-off F1 changes the env spawn between gens; absence of `transgenerational` block leaves loop behaviour byte-equivalent; pairing validator rejects mismatched configs at config-load time.
- [ ] 5.5 Also extend `tests/.../utils/test_config_loader.py` with ~4 cases covering the new Pydantic validators (decay range, schedule coverage, pairing validator).
- [ ] 5.6 Run pytest + smoke + pre-commit clean.

## 6. Config + campaign shell (Commit 6)

- [ ] 6.1 Create `configs/evolution/transgenerational_pathogen_avoidance_lstmppo_klinotaxis.yml`. Population 16, generations 4, brain `lstmppo` with klinotaxis sensors. Use the existing `predators:` YAML block with `predator_type: stationary`, `speed: 0`, and larger `damage_radius` (not a new `pathogen_lawns:` keyword — to minimise loader surface area, the M6 implementation reuses the existing predator-block schema directly; "pathogen lawn" is documentation vocabulary only). `transgenerational` block with `enabled: true`, `decay_factor: 0.6`, `extraction_seed: 424242`, and `lawn_schedule` covering F0 (pathogen on, ppo_train_episodes=50) + F1/F2/F3 (pathogen off, ppo_train_episodes=0). `inheritance: transgenerational`.
- [ ] 6.2 Create `scripts/campaigns/phase5_m6_transgenerational_lstmppo_klinotaxis.sh`. Multi-seed parallel wrapper (4 seeds: 42, 43, 44, 45). Subcommands `--smoke`, `--pilot`, `--full`. `--smoke` runs F0-only 1 seed × pop 6 ≈ 30 min for calibration. `--pilot` 1 seed × pop 6 × 4 gens. `--full` 4 seeds × pop 16 × 4 gens, paired TEI-on + TEI-off (override `transgenerational.enabled=false` + `inheritance=none` on the second arm).
- [ ] 6.3 Output directory convention: `evolution_results/m6_transgenerational/{tei_on|tei_off}/seed-{N}/<session_id>/`.
- [ ] 6.4 Tests: extend `tests/.../test_smoke.py` parametrize list with the new config so it gets a CLI-end-to-end smoke at `--runs 2 --theme headless --timeout 300`. ~1 added smoke case.
- [ ] 6.5 Run pytest + smoke + pre-commit clean.

## 7. Per-gen choice-index evaluator + paired-arm aggregator (Commit 7)

- [ ] 7.1 Create `scripts/campaigns/transgenerational_per_gen_eval.py`. Mirrors `scripts/campaigns/baldwin_f1_postpilot_eval.py`. Computes per-agent per-episode `choice_index = 1 - (steps_inside_damage_radius / total_steps)`. Per-generation: mean across all agents × all episodes. Outputs `per_gen_choice_index.csv` with columns `seed, arm, generation, agent_id, episode, choice_index`.
- [ ] 7.2 Create `scripts/campaigns/aggregate_m6_pilot.py`. Mirrors `scripts/campaigns/aggregate_baldwin_retry_pilot.py`. Sections: (a) pre-flight sanity checks (F0 calibration envelope, paired-seeds parity), (b) per-generation choice-index per arm × seed × generation, (c) decision-gate evaluation (F1 ≥40% × F0, F2 ≥25%, F3 ≥15%, monotone non-increasing) per seed, (d) cross-seed verdict aggregation (GO iff ≥2 of 4 seeds), (e) TEI-on vs TEI-off retention table, (f) wall-time summary, (g) decision-gate verdict markdown.
- [ ] 7.3 Add `pathogen_choice_index` alias column in aggregator output for vocabulary clarity (parallel to underlying `predator_avoidance`).
- [ ] 7.4 Tests: synthetic input fixtures verifying decision-gate logic (deterministic-pass case, deterministic-fail case, monotone-violation case). ~6 cases under `tests/...` if reusable as unit tests, otherwise document the aggregator's correctness via a comment-pinned reference fixture.
- [ ] 7.5 Run pytest + pre-commit clean.

## 8. F0 calibration smoke + pilot + full campaign + logbook (Commit 8)

Execution-only work below; no code lands in commit 8 except the logbook itself.

- [ ] 8.1 Run F0 calibration smoke (`--smoke`). Verify `0.45 ≤ mean F0 choice_index ≤ 0.85`. If outside envelope, STOP and report; retune `damage_radius` and `ppo_train_episodes` before unblocking M6.5.
- [ ] 8.2 Run pilot (`--pilot`): 1 seed × pop 6 × 4 gens, paired arms, ~4 wall-hours. Aggregator produces preliminary verdict.
- [ ] 8.3 Pause for user review (per the project's "logbook review before verdict" convention). User reviews preliminary evaluation outputs, confirms threshold calibrations and any pilot-driven gate refinement.
- [ ] 8.4 Run full campaign (`--full`): 4 seeds × pop 16 × 4 gens, paired arms, ~16 wall-hours. Aggregator produces per-seed + cross-seed verdict.
- [ ] 8.5 Pause for user review of full campaign outputs BEFORE finalising logbook.
- [ ] 8.6 Write `artifacts/logbooks/018-transgenerational-memory.md` (in `docs/experiments/logbooks/`). Cite Hunter critique + Murphy rebuttal + Vidal-Gadea independent validation. Pathogen-lawn-as-STATIONARY-predator methodology note. F0 calibration outcome. Per-seed retention table. Cross-seed verdict. Carry forward the M6 framing locked in `openspec/changes/phase5-tracking/tasks.md` line 255 ("first computational replication of the canonical (defended) TEI mechanism on an RL substrate").
- [ ] 8.7 Stash run artefacts under `artifacts/logbooks/018-transgenerational-memory/`. Audit file sizes vs `.gitattributes` LFS rules before staging; flag any >100 KB files not covered before committing.
- [ ] 8.8 Sanitise stashed artefacts for absolute home paths (`/Users/`, `/home/`, `C:\Users\`) before committing.
- [ ] 8.9 Update `openspec/changes/phase5-tracking/tasks.md` M6.1–M6.8 ticks. Update `docs/roadmap.md` Phase 5 milestone tracker M6 row with verdict.
- [ ] 8.10 Archive this OpenSpec change via `openspec-archive-change` after verdict published.
