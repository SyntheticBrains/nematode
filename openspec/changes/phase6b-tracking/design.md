## Overview

`phase6b-tracking` is the living tracker for Phase 6b — the deferred completion of Phase 6 after the 2026-07 Gate-3 sub-phase split. Phase 6b is the L3 NEAT topology search (T8) plus its synthesis addendum (T9b). This design records the small number of 6b-specific decisions and — deliberately — leaves the load-bearing NEAT-implementation choices as **open questions** to be resolved inside the per-milestone 6b OpenSpec change, because they depend on a not-yet-made environment-vectorisation decision.

## Goals / Non-Goals

**Goals:**

- Give the committed Phase 6b work an honest home (genuine `[ ]` not-started tasks) so `phase6-tracking` can archive cleanly at the Phase 6a synthesis rather than holding open on ticked-but-undone boxes.
- Record Phase 6b's preconditions and open design questions so a future session can resume without re-deriving them.

**Non-Goals:**

- Deciding the NEAT implementation (TensorNEAT integration specifics, env-vectorisation approach, population/generation budget) — those belong in the per-milestone 6b change once 6b starts.
- Re-opening co-evolution — it is deferred out of Phase 6 entirely (see Decision B).

## Design Decisions

### Decision A: Phase 6b inherits a fresh tracking change (not appended to `phase6-tracking`)

Per the `phase6-tracking` spec's *"Phase 6a / 6b sub-phase split triggered at Gate 3"* scenario, because Phase 6b contains more than one tranche (T8 NEAT + T9b synthesis), it **inherits a fresh `phase6b-tracking` change**. This lets `phase6-tracking` end with genuine `[x]` on everything it owns (T1–T7 + controls + validation + T9a) and archive at the Phase 6a synthesis, while the committed 6b work lives here as honest `[ ]` not-started.

The alternative — ticking the T8 tasks `[x]` "deferred" inside `phase6-tracking` — was rejected for the NEAT work because it is *relocated, not dropped*: it is committed, scheduled work that genuinely executes in 6b, so a fake tick would misrepresent it to `openspec archive`. (That house-style `[x]` + note *is* used for genuinely-dropped scope — see Decision B.)

### Decision B: Co-evolution is NOT a Phase 6b deliverable

The ex-T8.4 matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP co-evolution test (Phase 5 M5's architecture-asymmetry follow-up) is **deferred with no scheduled destination** — it is not reassigned to Phase 6b or any later phase. Rationale: co-evolution is too compute-intensive for the phase, and the M5 diagnosis was already independently corroborated (Resendez Prado 2026, "transparent regime" self-play suppression). It is recorded as a ticked scoping decision in `phase6-tracking` and in the roadmap (§ Research Questions RQ4); the lag-matrix cross-pairing instrument (Phase 5 logbook 017) is retained and reusable whenever a future phase commits to it.

### Decision C: The connectome-structure controls are Phase 6a, not 6b

The degree-preserving rewired-null control (Dhiman 2026) and the learnable-gap-junction variant are **PPO weight-search on the connectome topology**, not topology search. They need no TensorNEAT, GPU, or env-vectorisation, and reuse the existing T7 continuous pipeline. So they live under `phase6-tracking` as `T7.controls.*` (Phase 6a). Phase 6b's "is the connectome a local optimum?" analysis (T8.5) cross-references the 6a rewired-null result — the credible verdict is relative to the degree-preserving null, not to NEAT/MLP alone.

## Open Questions (resolved in the per-milestone 6b change, not here)

- **Environment vectorisation.** TensorNEAT's ~500× speedup over neat-python assumes a JAX-vmappable environment (Brax/gymnax). `Continuous2DEnvironment` is a Python/PyTorch sim and is not vmappable as-is, so **env throughput — not the GPU — is the binding constraint**. Open decision at 6b start: port the env physics to a vmappable form (JAX or batched-torch) for the NEAT eval loop, vs run a reduced-population / single-behaviour scoped search against the existing env. This decision gates the whole tranche's feasibility and compute budget.
- **Compute tier.** A single consumer GPU (e.g. RTX 2080, 8 GB) is sufficient for the NEAT *algorithm* (tiny networks) and for a **scoped** single-behaviour search, but not for a full-fat multi-behaviour NEAT campaign without env-vectorisation. Pin the population size, generation count, behaviour count, and seed count against the resolved env-vectorisation path.
- **Behaviour scope.** The Phase 6 exit criterion requires topology search on *at least one* behaviour; klinotaxis is the natural first. Whether 6b runs one or three behaviours is set by the resolved compute path.

## Risks

- **NEAT fails to learn the integrated lethal cell.** T4's FeedforwardGA collapsed to 0% (Logbook 025), reproduced at T7 (GA 15.0, Logbook 029) — a yellow flag for evolutionary search on this substrate. Genuine NEAT (topology + weights) is more capable than GA-on-fixed-topology, but budget for graded-fitness / curriculum staging, and lead with the cheaper 6a rewired-null control (which uses the proven PPO pipeline) so the connectome-structure verdict does not depend on NEAT converging.
- **Headline partly pre-answered.** Logbook 029 already shows a plain MLP beats the connectome, so "is the connectome a topological optimum?" is directionally *no* before NEAT runs. Frame 6b as confirmatory + the local-optimum characterisation, not as a result whose direction is in doubt.
