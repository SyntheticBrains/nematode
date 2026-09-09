# Design: structured, pathway-specific instruction

## What the wiring says

Computed from the vendored atlas and the Cook 2019 chemical graph, before anything was built:

| source | neurons | neurons they synapse onto |
|---|---|---|
| dopamine | 8 | 104 |
| octopamine | 2 | 31 |
| serotonin | 2 | 8 |
| **all three** | **12** | **123 of 302** |

Keying a synapse by its **post-synaptic** neuron, 2,024 of 3,709 chemical synapses (54.6%) are
instructed; keying by the pre-synaptic neuron gives 52.8%, and requiring both ends gives 36.9%.

The post-synaptic keying is the one this change uses, and the reason is mechanical rather than
aesthetic: a modulator gates plasticity at a synapse by acting on the cell that owns it, and in
this substrate the plastic weight `w[pre, post]` belongs to `post`'s incoming set — the same axis
homeostasis normalises over. Keying by the pre-synaptic neuron would gate a synapse by what its
*sender* is told, which no mechanism supports.

## The routing model

A synapse is **instructed** when its post-synaptic neuron receives chemical input from an
aminergic neuron. For instructed synapses the update is today's three-factor rule. For the rest,
the modulator is replaced by `1.0` — the **unmodulated Hebbian term**, which is the panel's other
registered floor, not a new invention:

```text
u = η · m · E / ρ_E   at instructed synapses
u = η · 1 · E / ρ_E   elsewhere
```

The arm is therefore an explicit interpolation between two arms every panel has already measured:
reward-modulated learning on 54.6% of the substrate and co-activity learning on the remaining
45.4%. That framing is deliberate. It means a result can be read against both floors rather than
against a novel rule, and it means the change introduces **no new hyperparameter** — the split is
read off the wiring, not chosen.

The alternative for uninstructed synapses is to freeze them (`m = 0`, no update at all). It was
considered and rejected: it would confound "credit is routed" with "45% of the substrate stopped
learning", and the panels already show that freezing a substrate changes its outcome distribution
on its own. A frozen-elsewhere variant is a clean follow-up if routing turns out to matter.

## An honesty note on the pathway model

**Aminergic transmission in *C. elegans* is substantially extrasynaptic.** Dopamine, serotonin and
octopamine act by volume release onto receptors expressed by neurons that need not be synaptic
partners of the releasing cell; the wired reach computed above is therefore a **lower bound and a
modelling choice, not the biology**. What this change tests is a specific, falsifiable proxy — that
the *synaptic* reach of the aminergic neurons picks out a functionally meaningful subset — and the
record must say so in those terms.

Two consequences follow and both are written into the test rather than left to interpretation.
First, a negative here does not refute structured instruction; it refutes this proxy for it. The
receptor layer (B.3) is what would replace wired reach with expressed-receptor reach, and Logbook
046 already promoted that layer to a prerequisite for the inhibitory-brake question. Second, the
instructed fraction is reported with the result, because a proxy that turned out to cover 5% or
95% of the substrate would make the arm uninterpretable in opposite directions, and the reader
should not have to take 54.6% on trust.

## Where it sits in the rule

The routing multiplies the modulator, so it is one elementwise factor inside the Hebbian term and
nothing else moves:

1. Hebbian term, trace-normalised, **with the modulator replaced by 1.0 off the instructed set**.
2. Weight decay; any decorrelating term; any consolidation term.
3. Mask, then write; Dale's-law projection; homeostatic rescale; clamp, last.

Under the unmodulated arm the modulator is already `1.0` everywhere, so routing is a no-op there —
which is correct, and is asserted, because an "unmodulated with routing" config would otherwise
look like a distinct arm while being the plain Hebbian floor.

## Configuration and derivation

| field | default | meaning |
|---|---|---|
| `third_factor` | `global` | either `global` or `pathway` |

`pathway` requires the substrate's transmitter identities, so it is refused where they are absent
— on the connectome config beside the existing sign validators, duplicated as a brain-construction
guard (`model_copy` skips validators), and outright on the dense MLP yardstick, which has no
aminergic neurons to route from. The mask is derived once at construction from the connectome and
the classification table, registered as a wiring buffer beside `chem_sign`, and reported through
the same weight-file guard, so a checkpoint saved under one routing model is refused by the other.

## Telemetry

Two quantities beside the existing plasticity series: the **instructed fraction** (constant per
build, reported so it is never assumed) and the **instructed share of the update's magnitude**,
measured on the effective update the way the decorrelation share now is. The second is what makes
"credit reached only the synapses the wiring instructs" a measurement: under `pathway` the
modulated part of the update must be confined to the instructed set, and the telemetry says
whether it was.

## The registered test

Four arms — pathway and global third factors on the wild-type and the rewired null — seeds 1–16
paired, the panel's 3000-episode plastic budget, the committed plateau-tail metric, and the
panel's registered extension of a fresh run at 1.5× for a run the plateau detector marks
non-converged. 64 runs. The global arms are **re-run concurrently** rather than read from panel 1's
committed table: that arm has 8 committed seeds at this budget and four panels have now found n = 8
too weak for this outcome's shape. Panel 1's values are reported beside the new global arm as a
consistency check, not as a comparator.

Four one-sided paired tests corrected together under BH-FDR at α = 0.05:

- **S1** wild-type pathway over wild-type global — *does routing help at all* (primary).
- **S2** rewired-null pathway over rewired-null global — *does it help without the real wiring*.
- **S3** wild-type over rewired null under pathway routing — the wiring contrast with routing.
- **S4** wild-type over rewired null under the global scalar — the same contrast without it, a
  concurrent replication of the panel's own primary at n = 16.

**Verdict**, in order: `insufficient_seeds`; then **`no_routing_effect`** when neither S1 nor S2
confirms — the outcome in which routing changes nothing and the global scalar was not the
limitation; then `routing_helps_both` (S1 and S2), `routing_helps_wild_type_only` (S1),
`routing_helps_rewired_only` (S2). S3 and S4 annotate and never decide.

`routing_helps_both` is the outcome that would matter least for the wiring hypothesis and is named
so it cannot be reported as a win: a routed third factor that helps the scramble as much as the
animal is a fact about having two learning regimes in one network, not about *C. elegans*
connectivity. The record states that reading in advance.

## What a result means

A confirmed S1 licenses the routed third factor as the rule's instruction model and makes it the
substrate for any further rule work; it does **not** license the 2×2 panel, which stays gated on
the clone assay, since routing credit and holding a policy are different capabilities.
`no_routing_effect` closes the global-scalar explanation of four panels' worth of null results and
leaves structured instruction resting on the receptor layer, which is where Logbook 046 already
placed the inhibitory-brake question — a convergence the 7a shipment decision should record.

## Alternatives considered

- **Freezing uninstructed synapses** — confounds routing with a 45% reduction in the plastic set;
  a follow-up if routing matters.
- **Routing by pre-synaptic neuron, or requiring both ends** — 52.8% and 36.9% respectively; the
  post-synaptic keying is the one a modulator acting on a cell implies.
- **Per-amine arms (dopamine only)** — a real question and the principled prior for a reward
  signal, deferred to keep this change to one family; the mask derivation is written to make it a
  configuration rather than a rewrite.
- **Reading panel 1's committed global values instead of re-running them** — cheaper by 32 runs
  and rejected on the sample-size grounds above.
