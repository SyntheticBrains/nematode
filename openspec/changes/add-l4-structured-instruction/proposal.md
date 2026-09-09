# Structured, pathway-specific instruction (7a-ii B.4b)

## Why

Every panel so far has run the third factor as a **global scalar**: one reward-prediction error
broadcast to every plastic synapse. That is the registered baseline and it has never separated the
wild-type wiring from its scramble. The standing argument against it — from the roadmap's
reframing and from the electrosensory-lobe connectome, where wiring decides which synapses receive
instructive input (Perks et al., *Nature*, 2026-09-02) — is that credit assignment is a property of
the circuit, not a number in the air. This change routes the third factor through the wiring that
carries it and tests whether that matters.

The atlas makes the routing derivable for the first time. It identifies **12 aminergic neurons**
— 8 dopaminergic (`ADEL/R`, `CEPDL/R`, `CEPVL/R`, `PDEL/R`), 2 serotonergic (`NSML/R`) and 2
octopaminergic (`RICL/R`) — and Cook 2019 says who they synapse onto: **123 of 302 neurons**,
covering **2,024 of 3,709 chemical synapses (54.6%)** when a synapse counts as instructed by its
post-synaptic neuron. That is the property this test needs and the previous rung lacked. The
decorrelation test failed because a transmitter-only atlas grounds just 5.8% of synapses as
inhibitory — too few to build a brake from. Here the same metadata yields a subset that is
selective without being negligible, so a routed third factor is a real intervention rather than
either a no-op or a relabelling of the whole substrate.

Ratified with Chris 2026-09-09: four arms at n = 16, with the global-scalar comparator **re-run
concurrently** rather than read from panel 1's committed table, because that arm has only 8
committed seeds and four panels have now found n = 8 too weak for this bimodal outcome.

## What Changes

- **A derived instructive pathway** on the connectome substrate: the set of neurons receiving
  chemical input from an aminergic source, computed from the vendored atlas and the Cook 2019
  wiring, exposed as a per-synapse boolean mask and reported as a fraction so a build that routes
  everything or nothing is visible immediately.
- **A routed third factor** (`third_factor: pathway`): instructed synapses see the modulator;
  uninstructed synapses fall back to the **unmodulated Hebbian term**, which is the panel's other
  registered floor. The arm is therefore an explicit interpolation between two things the panels
  have already measured — reward-modulated learning on 54.6% of the substrate, co-activity
  learning on the rest — and not a new free parameter. `global` stays the default and is
  byte-identical.
- **Telemetry**: the instructed fraction and the share of the update's magnitude carried by
  instructed synapses, so "credit reached only the instructed set" is a measured claim rather than
  an assumed one.
- **The registered test**: pathway and global third factors on both wirings, seeds 1–16, the
  panel's 3000-episode plastic budget, as a four-test BH-FDR family whose first substantive
  outcome is that routing changes nothing.
- Configs, a harness, a launch record, the runs, records under
  `supporting/047-l4-structured-instruction/`, tests, docs.

Out of scope: receptor-class gating (B.3 — see the honesty note in the design), diffusible-signal
dynamics, consolidation, and the 2×2 panel re-run, which remains gated on the clone assay.

## Capabilities

**Modified**: `connectome-substrate` (the derived aminergic pathway), `learning-rules` (the routed
third factor and its telemetry), `l4-plasticity-panel` (the registered test).

## Impact

- New: the pathway derivation and its mask, config fields, four arm configs, an analysis harness,
  the supporting directory. Edited: `connectome/` loader surface,
  `brain/arch/_plasticity_config.py`, `brain/arch/connectome_ppo.py`,
  `learning_rules/three_factor.py`, `docs/architectures.md`, `CHANGELOG.md`.
- Defaults are byte-identical: `third_factor: global` takes today's code path and derives no mask.
