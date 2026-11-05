## Context

- **Benchmark v4** is VALID and the default; C.1 (v3)'s answer fails it at 0 of 5 on the on-lawn onset rate ([Logbook 124](../../../docs/experiments/logbooks/124-benchmark-v4.md)).
- **The probes** (2026-11-05, discarded seeds 620–646 M, CMA 9441–9448): each is one sep-CMA restart against v4's ranking score with C.1b's doubly guarded stop, its answer graded on v4 with presence on 5 populations. The table is in Logbook 125.

## Goals / Non-Goals

**Goals:** say whether C.1's model can reach v4 and what it lacks; keep the one mechanism that answers it, off by default; record every probe.

**Non-Goals:** a registered fit (deferred by the maintainer); changing the benchmark; the dropped candidates.

## Decisions

1. **`r_dwell_on_food`** (`BatchedMechanistic`): a dwelling animal's reversal rate wherever food contact is above 0, in place of `r_dwell`. Off food, and while roaming, the state's rate applies; the klinokinesis, edge and search terms multiply it as before. It is off unless given, and the reversal hazard's product is computed in the same order, so without it every output is byte-identical: `speedup_identity_check.py check --build build/c1-v4/identity` against a baseline recorded on the code before any candidate, 59 of 59 jobs.
2. **Why this form.** A state-free off-food rate (probed) removed local search's decay, which C.1 makes from the two states' different rates off food; raising `r_dwell` (sensitivity-tested) breaks local search and Chen's chemotaxis. The on-food dwelling rate moves no off-food statistic in its sensitivity test.
3. **Not kept:** the reversal–speed coupling and intermittent dwelling, each failing the onset rate (z −3.7 to −4.2). Their code is at `886e7bfc`, where their probes reproduce. The off-food rate's code is recorded in its probe's script.
4. **The refit is deferred,** with the drafted design parked in `parked_fit_design.md`. Its open question is the start: every box-centre run failed, and a warm start from C.1 (v3)'s answer needs a D38 amendment.

## Risks / Trade-offs

- **[C.4 waits on a refit]** → C.1 enters C.4 from a refit run when C.4 approaches; until then C.2's informal yardstick is the warm probe's D (1.28–1.44), as description.
- **[The mechanism was chosen on the benchmark]** → Disclosed, with every candidate's record; C.1's mutant predictions remain its held-out test.
