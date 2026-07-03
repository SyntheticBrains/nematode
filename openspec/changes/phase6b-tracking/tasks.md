# Tasks: Phase 6b (L3 NEAT Topology Search + Synthesis) Tracker

This is the living checklist for Phase 6b — the deferred completion of Phase 6 after
the 2026-07 Gate-3 sub-phase split. Phase 6b = Tranche 8 (L3 NEAT topology search) +
Tranche 9b (synthesis addendum + Phase 6 COMPLETE marker). The Phase 6a tracker
(`phase6-tracking`) owns T1–T7 + the connectome-structure controls + T9a, and archives
at the Phase 6a synthesis.

**Status legend**: `[ ]` not started, `[x]` complete.

**Preconditions (all three MUST hold before T8 execution begins):**

1. **Gate 3 GO** — Phase 6a closed (T7 ranking + real-worm validation + connectome-structure controls in hand).
2. **GPU availability** — a card for the TensorNEAT campaign (a single consumer GPU suffices for a *scoped* search; see design.md § Open Questions § Compute tier).
3. **Environment-vectorisation decision** — the binding constraint. TensorNEAT's ~500× speedup assumes a vmappable env; `Continuous2DEnvironment` is not one yet. Resolve vmappable-env-port vs reduced-population/scoped-search **before** committing the GPU campaign (design.md § Open Questions § Environment vectorisation).

> **Not in Phase 6b — co-evolution.** The ex-T8.4 matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP co-evolution test is **deferred with no scheduled destination** (design.md § Decision B; roadmap § Research Questions RQ4). It is NOT a Phase 6b task. The lag-matrix instrument (Phase 5 logbook 017) is retained for whenever a future phase commits to it.

<!-- -->

> **Coarse-grained by design.** These sub-tasks are the load-bearing shape; the per-milestone 6b OpenSpec change (`add-l3-neat-topology-search`, authored when 6b starts) elaborates them against the resolved env-vectorisation decision.

## Tranche 8 — L3 NEAT Topology Search on Upgraded Substrate

**OpenSpec change**: `add-l3-neat-topology-search` (placeholder; created at 6b start)
**Status**: 🔲 not started (gated on the three preconditions above)
**Roadmap layer**: L3
**Approx duration**: 6-10 weeks
**Dependencies**: Phase 6a closed (Gate 3 GO)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Phase 6a/6b split + § The layered platform § L3

- [ ] T8.0 **Env-vectorisation decision (the real prerequisite).** Decide and document: port `Continuous2DEnvironment` physics to a vmappable (JAX or batched-torch) form for the NEAT eval loop, vs a reduced-population / single-behaviour scoped search against the existing Python env. Pin the resulting population / generation / behaviour / seed budget. Blocks the GPU campaign.
- [ ] T8.1 Integrate TensorNEAT (GPU-accelerated NEAT, JAX/vmap; ~500× speedup over neat-python — GECCO 2024 Best Paper, Wang et al.; ACM TELO 2025). *Operational caveats: no semver releases — **pin a commit hash**; couples to specific Brax/gymnax versions — budget for the source-install path. No newer GPU-NEAT library supersedes it.*
- [ ] T8.2 NEAT topology + weight evolution on the L1 plugin interface from T2. The plugin should accommodate NEAT-evolved topologies as natively as fixed-topology architectures (the T2 topology/rule factoring pays off here).
- [ ] T8.3 Topology-vs-connectome head-to-head on at least one behaviour on the upgraded continuous substrate (klinotaxis is the natural first).
- [ ] T8.5 Cross-architecture analysis: "is the wild-type connectome a local optimum?" Cross-reference the T7 connectome ranking (Logbook 029: connectome 5th of 6) with the T8 NEAT-evolved baseline **and the Phase 6a `T7.controls.rewired_null` degree-preserving control** — the verdict is only credible relative to the degree-preserving null, not to NEAT/MLP alone (Dhiman 2026). *Yellow flag: T4/T7 GA collapse (0% / 15.0) means evolutionary search struggles on the integrated lethal cell; genuine NEAT is more capable than GA-on-fixed-topology, but budget for graded-fitness / curriculum staging.*
- [ ] T8.6 Update `docs/roadmap.md` Phase 6 Tranche Tracker T8/6b row + this checklist.
- [ ] T8.7 Publish the 6b logbook (suggested: `docs/experiments/logbooks/0XX-l3-neat-topology-search.md`).

**T8 risk-mitigation pivot (per roadmap)**: if NEAT-evolved topologies and the wild-type connectome converge to indistinguishable performance, that *is* the finding — "the connectome is competitive with evolved topologies on these behaviours." The optimal-primary framing weakens; the connectome-primary framing strengthens. Acceptable outcome; pivot the headline framing if it lands.

*(T8.3b / T8.3c from the original Phase 6 plan are NOT here — they are the connectome-structure controls, which moved to Phase 6a as `T7.controls.rewired_null` / `T7.controls.learnable_gj`. T8.4 is the co-evolution test, deferred with no destination — see the banner above.)*

## Tranche 9b — Phase 6b Synthesis Addendum

**OpenSpec change**: `add-phase6b-synthesis-logbook` (placeholder; created at 6b close)
**Status**: 🔲 not started
**Roadmap layer**: synthesis
**Dependencies**: T8 closed
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Phase 6 exit criteria

- [ ] T9b.1 Append the NEAT topology-search result (T8) to the Phase 6a synthesis; cross-reference the 6a `T7.controls.rewired_null` control for the "is the connectome a local optimum?" verdict. Document negative findings honestly (Phase 5 precedent) if NEAT came back STOP.
- [ ] T9b.2 Update `docs/roadmap.md`: Phase 6 status → ✅ COMPLETE; record the L3 exit-criterion outcome and Phase 6's terminal verdict; flip the Tranche Tracker T8/9b rows to their terminal state.
- [ ] T9b.3 Publish `docs/experiments/logbooks/0XX-phase6b-synthesis.md` (or fold into the T8 logbook if slim).

> Archiving `phase6b-tracking` is an operator-side step that does NOT block task completion (an "archive me" task would self-block, since `openspec archive` requires every task ticked — same precedent as `phase5-tracking` and `phase6-tracking`). The archive happens after the 6b synthesis PR merges, via `openspec archive phase6b-tracking`.
