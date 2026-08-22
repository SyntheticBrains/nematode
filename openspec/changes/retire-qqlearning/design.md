## Context

`QQLearningBrain` (hybrid quantum-classical Q-learning with experience replay,
782 lines) was one of the six architectures shipped in Phase 0. The roadmap closed
it at `docs/roadmap.md:1216` — *"evaluated, not competitive; deprioritised"* — but
the code was never retired, so it still self-registers, still appears in the
architecture count, and still has to be reasoned about by every cross-brain change.
The WS2 `_policy.py` consolidation had to classify it (as non-policy-gradient, and
therefore out of scope); Phase 7's L4 plasticity work would have to do the same.

Meanwhile the module itself is visibly unfinished: 9 of the package's 11 `TODO`s,
two dead parameters, and a `copy()` that silently drops the entire learning state
on a code path many-worlds mode actually calls.

The decision to retire rather than repair was taken by the maintainer on
[#282](https://github.com/SyntheticBrains/nematode/issues/282).

## Goals / Non-Goals

**Goals:**

- Remove the brain and every live reference, leaving the tree consistent: registry,
  enum, factory, config union, counts, docs.
- Keep the removal loud for anyone with an out-of-tree config that names it —
  an unknown brain type should fail at config load, not fall back to something else.
- Leave the historical record intact: logbooks describe what existed at the time
  and are not edited.

**Non-Goals:**

- Not a change to any other brain's behaviour, and not an opportunity to touch the
  registry mechanism itself.
- Not a decision about the other three non-policy-gradient brains (`mlpdqn`,
  `qvarcircuit`, `feedforwardga`). Each has its own standing; this change is about
  the one the roadmap already closed.
- Not a repair of the `copy()` defect — retiring the brain removes the need.

## Decisions

### D1 — Retire, not mark-dormant

Issue #282 offered three options: retire, mark dormant and fix only the honesty
problems, or complete it. **Retire** was chosen.

The argument for dormant-and-fix is that the code still compiles and has 30 tests,
so the cost of keeping it looks low. It is not: the cost is paid by *other* work.
Every change that sweeps across brains — the `_policy.py` consolidation, the
upcoming L4 plasticity pass, any registry or config-schema change — has to decide
what this brain needs, and the honest answer each time is "nothing, it is not
used". That decision is cheaper made once, permanently.

The argument against retiring is loss. There is little to lose: git history keeps
the implementation, and the evaluation that produced the "not competitive" verdict
is recorded in the roadmap and Logbook 023's inventory. Nothing is destroyed.

### D2 — No tombstone module, unlike `remove-nematodebench`

The NematodeBench removal kept `openspec/specs/benchmark-management/spec.md` as a
retired-capability tombstone, because an entire capability disappeared and a reader
arriving from an old link needed to find out where the live parts went.

Nothing analogous applies here. `configuration-system` survives — it is amended,
not retired — so there is no spec to tombstone, no requirement to redirect, and no
404 to prevent. Adding a placeholder module would be dead code of exactly the kind
the recent cleanup has been removing.

**Consequence, stated:** an out-of-tree config with `brain: qqlearning` will fail at
config load with an unknown-brain-type error. That is the intended behaviour — the
alternative is silently substituting a different architecture, which would be far
worse — and there are no in-repo configs to migrate.

### D3 — Backward compatibility was checked, and does not bite

The `remove-nematodebench` change had a real compatibility hazard:
`composite_benchmark_score` had to keep its name because 421 historical artifacts
carried the key and `ExperimentMetadata.from_dict` ends in `cls(**data)` with
Pydantic's default `extra="ignore"`, so a rename would have read back as `None`
without raising.

The equivalent check here comes back clean:

- `ExperimentMetadata.brain_type` is typed `str`
  ([`metadata.py:491`](../../../packages/quantum-nematode/quantumnematode/experiment/metadata.py)),
  **not** the `BrainType` enum — so a historical record naming `qqlearning` still
  loads after the enum member is gone.
- There are **0** artifacts under `artifacts/` carrying the string at all.

So no field keeps a legacy name for compatibility, and no artifact needs migrating.
This is recorded because the absence of the hazard is the non-obvious part.

### D4 — Fix the stale legacy-alias list in the same requirement

The `configuration-system` requirement being amended reads:

> **AND** existing "qvarcircuit", "qqlearning", "mlpreinforce", "mlpppo", "mlpdqn"
> types are also valid (legacy aliases: "modular", "qmodular", "mlp", "qmlp")

Removing `qqlearning` from that sentence means editing it anyway — and the rest of
the sentence is also wrong. **None of `modular`, `qmodular`, `mlp`, `qmlp`, `ppo`
exists in the code**: they appear in no `BrainType` value and in no alias mapping in
`config_loader.py`. They were presumably removed at some point without the spec
following.

Leaving a clause known to be false inside a requirement I am already rewriting
would be worse than fixing it. The delta therefore states the valid types by
reference to the registry rather than by re-enumerating a list that has drifted
twice, so it cannot drift a third time.

*Alternative rejected:* amend only the `qqlearning` mention and leave the aliases.
That preserves a known-false obligation in a live spec for tidiness reasons.

### D5 — The architecture count is derived, then written down

Four files state the count (`README.md`, `AGENTS.md`, `CONTRIBUTING.md`,
`openspec/config.yaml`) and `openspec/config.yaml` also enumerates every name. WS4
found that enumeration had already drifted (it said 25 and listed 25 while the
registry held 27).

So the count and the enumeration are re-derived from the live `_REGISTRY` and
verified programmatically as part of the tasks, not hand-edited from 27 to 26.

## Risks / Trade-offs

- **\[An out-of-tree config or notebook names `qqlearning`\]** → it fails loudly at
  config load rather than silently substituting. No in-repo config does; the failure
  mode is the desired one, and the removal is noted in the roadmap for anyone who
  hits it.
- **[Losing a quantum baseline for future comparison]** → the evaluation that
  retired it is already recorded, and git history keeps the implementation. If a
  quantum Q-learning arm is ever wanted again it would be rebuilt against the
  current L1 plugin interface rather than resurrected from a 782-line module with
  nine TODOs.
- **[The removal misses a reference and breaks at runtime rather than at import]**
  → the registry/enum consistency guard (`assert_registry_matches_enum`, invoked
  from the package `__init__`) fails fast on a mismatch, and the tasks include a
  grep audit with a declared expected answer, following the WS2 precedent where an
  open-ended sweep missed a site.
- **[Scope creep into the other three non-PG brains]** → explicit non-goal. Each has
  its own standing; only this one has a roadmap line closing it.

## Migration Plan

1. Delete the module and its test; remove the enum member, `BRAIN_TYPES` entry,
   registry export, factory branch and config union entry. Run the consistency
   guard.
2. Re-derive the count and the `openspec/config.yaml` enumeration from the live
   registry; update the four count sites and the two docstring mentions.
3. Land the `configuration-system` delta (including the D4 alias correction) and
   record the retirement in the roadmap and `OPTIMIZATION_METHODS.md`.
4. Grep audit against a declared expected answer; full gate; archive.
