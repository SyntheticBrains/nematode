# A decorrelating term for the local rule (7a-ii)

## Why

Logbook 044 grounded the substrate's synapse signs in the neurotransmitter atlas and found that
reward-free Hebbian learning got **substantially worse**: the wild-type arm fell from 31.5 to
14.0 and the rewired null from 17.4 to 9.1, both intervals clear of zero. It also issued a
falsifiable prediction — on a grounded, mostly-excitatory substrate, a rule with an anti-Hebbian
or decorrelating term should recover what the purely potentiating rule loses. This change builds
that rule and runs the test.

Recon sharpens the mechanism the logbook stated in outline. The eligibility trace is
`outer(prev_h, h)` with `h = tanh(preact)`, so it is **sign-carrying**: the rule potentiates a
synapse where pre- and post-synaptic activity agree in sign and depresses it where they disagree.
Under random signs half the synapses were inhibitory, and the anticorrelation an inhibitory
synapse produces is what depressed it further — a self-limiting loop that acted as the network's
brake. Grounded, the network is 80% excitatory and that loop is mostly gone; what remains is
positive feedback checked only by homeostasis and the bound. The same telemetry shows the rule
fighting the biology rather than using it: left free it ends with 15% of grounded synapses
carrying the opposite sign to their transmitter, and under Dale's law it drives 13% of them to
exactly zero instead.

Consolidation was screened first and none of its three mechanisms held a cloned competent policy
(Logbook 045), with the reading that slowing the update is not the same as consolidating a
policy. This change asks the other question the panels left open: not whether the rule stops, but
whether it has the *right* update to begin with on a substrate whose signs are real.

Ratified with Chris 2026-09-09: the decorrelating variant ahead of structured routing, tested by
re-running Logbook 044's own Hebbian protocol rather than the clone assay, since the clone assay
screens a mechanism's ability to hold a policy and this prediction is about learning from random
weights.

## What Changes

- **Anti-Hebbian inhibitory plasticity** (`decorrelation: anti_hebbian_inhibitory`): the update's
  sign is flipped on synapses the atlas grounds as **inhibitory**, so co-activity strengthens what
  an inhibitory synapse does rather than unwinding it. This is the mechanism the atlas unlocked —
  before B.1 no synapse had a transmitter identity to key on — and it is the arrangement the
  electrosensory-lobe connectome reports, where anti-Hebbian depression sits at specific,
  identified sites rather than everywhere (Perks et al., *Nature*, 2026-09-02). **It has no
  hyperparameter**: the term is the existing update with one factor of −1 on an identified subset.
- **Oja decorrelation** (`decorrelation: oja`): the substrate-general alternative, subtracting
  `η · γ · y² · w` per post-synaptic unit from the update — the classic term that removes runaway
  growth and decorrelates a unit's inputs without needing any sign information. One pinned
  coefficient.
- **Signs reach the rule whenever they are grounded**, not only when Dale's law is enforced, since
  the anti-Hebbian variant keys on them without constraining them.
- **Telemetry**: the share of the update's magnitude carried by the decorrelating term, so an arm
  that recovered by decorrelating is distinguishable from one that recovered by updating less.
- **The registered test**: Logbook 044's grounded Hebbian protocol re-run under each variant on
  both wirings — seeds 1–16, 1000 episodes — against 044's committed grounded per-seed values, as
  a four-test BH-FDR family with a verdict map that can say the prediction failed.
- Configs, a harness, a launch record, the runs, records under `supporting/046-l4-decorrelation/`,
  tests, docs.

Out of scope: structured pathway-specific instruction (B.4b), the receptor layer (B.3), any
consolidation mechanism, and the 2×2 panel re-run, which remains gated on the clone assay.

## Capabilities

**Modified**: `learning-rules` (the two decorrelating terms and their telemetry),
`connectome-ppo-brain` (signs supplied to the rule whenever grounded),
`l4-plasticity-panel` (the registered recovery test).

## Impact

- New: two code paths in `learning_rules/three_factor.py`, config fields on the plasticity mixin,
  four arm configs, an analysis harness, the supporting directory. Edited:
  `brain/arch/_plasticity_config.py`, `brain/arch/connectome_ppo.py` (the sign hand-off),
  `docs/architectures.md`, `CHANGELOG.md`.
- Defaults are byte-identical: `decorrelation: none` is today's code path and adds no operation.
