# Design — plastic rewired-null arm

## Context

The smallest arm in the shipment and the one the hypothesis turns on. No mechanism is added; the work is establishing, precisely, what is held constant between the two plastic wiring arms and proving it — because the primary contrast is worth exactly as much as that list is true.

## Decision 1 — no new mechanism, by construction

The rewiring transforms the `Connectome` object before the topology is constructed. Everything downstream is therefore built on the rewired graph: the strict mask `m_chem`, the support of `w_chem`, the gap-junction buffer, and — since the eligibility trace is masked by `m_chem` on every update — the trace's support and every plastic update. The plastic rule needs no awareness that it is running on a null; it reads the seam, and the seam is the rewired substrate.

This was confirmed rather than reasoned: after training the rewired arm, every changed weight and every non-zero trace entry lies on the rewired edge set.

## Decision 2 — what "matched" means between the two plastic arms

At one run seed, established by probe before being specified:

| Identical | Differs |
|---|---|
| motor readout (anatomical, RNG-free) | chemical edge mask `m_chem` |
| sensory-projection gains | placement of initial `w_chem` draws |
| action-noise parameters (`log_std`) | gap-junction matrix `g_gap` (also rewired) |
| trace configuration and plasticity hyperparameters (shared mixin) | |
| chemical edge count (3,709), every in- and out-degree, gap degree | |
| per-neuron initialisation scale `1/√(chemical in-degree)` | |

Why the periphery is identical: the rewiring consumes a **dedicated** generator, and it preserves the synapse count, so the brain's own RNG stream is asked for the same number of draws in the same order and arrives at the readout and gains in the same state. The readout is anatomical and consumes no randomness at all. Motor pools and sensory targets are defined by neuron *identity*, which rewiring does not touch — so the anatomical readout and the sensory injection mean the same thing on the null wiring as on the wild-type, which is what makes the contrast about wiring rather than about where the world enters or leaves the network.

**A claim that looked true and is not.** Per-neuron initial weight *energy* (the sum of squared incoming weights) is not preserved: the same sequence of normal draws lands on different pre/post pairs, so realised per-neuron sums differ. Only the per-neuron *scale* is preserved, because in-degree is. The probe caught this before it reached the spec; the requirement states scale, not energy.

## Decision 3 — pairing is by run seed and needs no protocol addition

`rewire_seed` defaults to the run seed, so seed *k* of the rewired arm is paired with seed *k* of the wild-type arm and the rewiring is deterministic under it. The panel's paired-seed design (n ≥ 8, BH-FDR) inherits this directly. A fixed single rewiring across all seeds was considered and rejected: it would make the null one *specific* scramble rather than the null *family*, and any effect would be confounded with that one draw.

## Decision 4 — primary arm only (ratified with the user)

D10 names one rewired arm. Rewired versions of the frozen and unmodulated floors would make the design fully factorial and are each a one-key config, but they add two n ≥ 8 arms to every panel run and answer a question — does rewiring move the untrained baseline? — that the panel's analysis plan has not yet said it will ask. The right moment to decide that is A.7's pre-registration, where the arm set and the tests are fixed together before any run. Deferring keeps this change to the contrast the hypothesis is built on and keeps a scope decision inside the change that owns it.

## Decision 5 — strict masking is inherited and required

The plastic parent runs `chemical_mask_mode: strict`, and the null arm inherits it. Under the plastic rule the update is masked regardless of mode, but strict is still the right statement: it makes the *forward* obey the null edge set as well as the update, so the arm is the null wiring in every respect and not merely in what learns. The config does not restate the key; the minimal-delta test guarantees it is inherited unchanged.

## Risks

- **The two plastic arms could be indistinguishable for a trivial reason** — both at the frozen floor. The floors already ship and are the detector; a null-by-incapacity reading is recorded in advance in the A.3 design and applies unchanged here.
- **A single seed's rewiring could be pathological.** The 034 mechanism rejects self-loops and duplicates and does not silently reseed; the panel's n ≥ 8 is what makes any one draw non-decisive.
- **Nothing here is a result.** The arm exists so the panel can run; no performance or wiring claim is made or implied.
