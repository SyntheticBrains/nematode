# Evaluate the node-perturbation endpoints with the perturbation off (7a-ii I.1c, step 0)

## Why

The clone assay measured the node-perturbation arm **under** its own perturbation. It scored 12.0
against the committed frozen clone's 38.7, and its frozen control — the same σ = 0.2 jitter with
no weight ever written — scored 8.9
([050 records](../../../docs/experiments/logbooks/supporting/050-l4-perturbation-clone-assay/details.md)).
That reading settled that the perturbation dominates the loss. It did not, and could not, say what
the policy the rule *left behind* is worth, because nothing has ever run those endpoint weights
with the perturbation removed.

On the MLP control the same question has an answer that changes the reading entirely: a policy
trained at σ = 0.2 scores 86% of the floor-to-optimum gap under that perturbation and **93%** with
it switched off. If the connectome endpoints behave the same way, the tension I.1 exposed — the σ
that makes the rule learn is the σ that makes a competent policy unrunnable — is an artefact of
evaluating under exploration noise, and every gate built on lowering σ asks the wrong question. If
they do not, the rule rewrote the policy (their cosine to the clone is 0.70) and the I.1 fail is a
genuine retention fail.

Eight frozen runs decide which. They cost under an hour, less than any alternative, and the next
registration is authored knowing the answer rather than carrying both branches.

## What Changes

- **One arm**: the committed comparator's own config — the wild-type plastic-set frozen clone,
  which runs no perturbation and writes no weight — with `weights_path` pointing at the eight I.1
  endpoints instead of the eight clones. A single-key delta from the comparator, as the comparator
  is from its parent. Budget 2000 episodes and the plateau-tail full-clear metric, matching every
  arm the assay has scored; the comparator's per-seed values are the ones committed.
- **The endpoints staged** under a `{seed}`-templated path beside the clones, with the source
  export of each recorded, as the clones were.
- **The pass rule is the assay's, unchanged**, applied to this arm; **both outcomes and what each
  licenses are written here, before the run.**
- Records under `supporting/052-l4-endpoint-evaluation/`; registry entry; docs.

Out of scope: any panel; any σ or rate change; any new mechanism. This change runs no learning at
all.

## Capabilities

**Modified**: `plasticity-evaluation` (a perturbing rule's endpoint is evaluated with the
perturbation off, and that evaluation gates what is registered next).

## Impact

- New: one config, the supporting directory, one registry entry in
  `scripts/analysis/l4_consolidation_screen.py` and its test pin. No package code changes.
- The outcome selects between two follow-on registrations, both named in the design; neither is
  authored until this has run.
