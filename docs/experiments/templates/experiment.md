# NNN: [A Title That States the Finding, Not the Topic] ([Phase / milestone])

<!--
Model on logbooks 070 and 071 for a registered campaign, 069 for a phase synthesis.
Every figure below must be re-derivable from the committed files listed under Artefacts.
-->

**Status**: `completed` | `active` | `abandoned` — **[the verdict in one clause]**. \[What is
established, what is unresolved at the panel's sensitivity, and any condition that now attaches to
an earlier result.\]

**Date**: YYYY-MM-DD. *\[If anything was re-run or corrected after first publication, say so here and
point to the corrections section.\]*

**OpenSpec change**: `change-name` ([capability it extends, and the requirement it adds, if any]).

**Pre-registration**: [supporting/NNN-slug/launch.md](supporting/NNN-slug/launch.md),
written before any panel seed ran. [Or: state plainly that none exists.]

## Objective

What question this answers, and why it matters now — the decision, earlier result or published claim
that makes it the next thing to run.

## Method

Design in brief; the launch record holds the full version. Name the factors, the cell, the arms,
the seeds, the run count and the wall clock. State the **primary** as registered — an interaction
where a manipulation is crossed with a structure contrast — and the metric rule, the minimum effect
in both directions, and the family correction. Name the instruments and whether they ran unmodified.

## Results

### [The registered gate]

Whether the baseline reproduced what it had to before anything else is read.

### [The primary]

| condition | effect | interval | p or q | reading |
|---|---|---|---|---|
| ... | ... | ... | ... | survival / unresolved / ... |

Read each row against the registered branches. A reading whose interval spans the bar is
**unresolved at this panel's sensitivity**, not null — give the sensitivity beside it.

### [Anything reported beside the primary]

## Corrections made in the open

Every defect found, every mid-course change to the protocol, and what each did to the numbers.
[Delete this section only if there were none, and say so.]

## What this establishes, and what it does not

**Establishes.** ...

**Leaves unresolved.** ...

**Does not establish.** ...

## Registered consequences

Each with one status: *met*, *unmet-with-reason*, *deferred-with-destination* (name it, in the
tracker and the roadmap), *superseded-by-result*, or *unreachable-with-reason*.

- ...

## Artefacts

Under [supporting/NNN-slug/](supporting/NNN-slug/):

- `launch.md` — the registration.
- `per-seed.csv` — [rows; the per-seed unit the primary test consumes].
- `analysis.json` — [what the headline figures are derived from].

Drivers: `scripts/analysis/...`. Raw campaign logs: archived off-repo.
