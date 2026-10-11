## Why

B.3 asks whether serotonin and PDF on the connectome gate roaming and dwelling as they do in the worm
(Flavell et al. 2013; Ji et al. 2021). Stage 0 (Logbook 089) gave the baseline: without modulators,
the connectome dwells on D.1's lawn cell where its untrained policy does not.

The connectome's default dynamics carry no state from one step to the next. Real roaming and dwelling
persist for minutes because these modulators are slow. So the modulator field is the connectome's
first across-step state, as well as its first neuromodulation.

The design was agreed with the user on 2026-10-11. The circuit is vendored from verified sources.

## What Changes

- **A verified circuit,** in `data/neuromodulators/` with provenance:
  - serotonin is released by NSM and inhibits AIY, ASI and RIF through MOD-1;
  - PDF is released by AVB, PVP and SIAV and excites AIY, RIM and RIA through PDFR-1;
  - both decay with a 60 s timescale.
- **A modulator field on the connectome PPO brain,** off by default and byte-identical when off:
  - **Concentrations:** two global concentrations, updated each step from their release neurons'
    activity, decaying with a fixed 60 s timescale.
  - **Gating:** a sign-fixed current on each receptor cell at every settling iteration, with a
    learnable strength per receptor.
  - **Food:** NSM receives a held current proportional to the last move's bite as a fraction of a
    full bite.
  - **PDF inhibits NSM** (Ji 2021), as a functional link, since NSM is wired only to the pharynx.
  - **Replay:** the concentrations are carried with the buffered states, so PPO's replay sees them.
- **Knockouts.** A config switch removes serotonin (*tph-1*) or every PDFR-1 effect (*pdfr-1*). It is
  applied at evaluation for the primary reading, and from the start of training for the raised
  mutants beside it.
- **Food ingestion reaches the brain:** `BrainParams.food_ingested` and `max_food_ingested`.
- **Grounded signs held under PPO.** With `enforce_synapse_signs`, PPO clamps each atlas-grounded
  synapse back to its sign after every step. The flag was previously read only by the plastic rules
  and was silently ignored under PPO.
- **A registered panel** on the sign-grounded substrate:
  - **Gates:** the modulated connectome learns, and dwells significantly more than its floor.
  - **The primary reading:** its acute *tph-1* and *pdfr-1* knockouts on mean on-lawn speed,
    oriented to the published directions (*tph-1* roams more, so faster; *pdfr-1* dwells more, so
    slower). The minimum is a third of real worms' dwelling-to-roaming speed gap.
  - **Beside it:** the dwelling share, raised mutants, a satiety link, strengths and concentrations.
- **Why the panel changed after the first pilot** (Decisions G and H). Random signs left every
  knockout's direction arbitrary. Policies sat in one state per seed, which the classified dwelling
  share cannot shift both ways. The random-sign arms and pilot are kept as that record.

## Capabilities

**Modified**:

- `connectome-ppo-brain`: the modulator field, its knockouts, and its persistence across steps and
  in replay; grounded signs held under PPO.
- `brain-architecture`: `BrainParams.food_ingested` and `max_food_ingested`.

## Impact

- `brain/arch/connectome_ppo.py`: the field, gating, release, ingestion drive, knockouts and replay.
- `learning_rules/ppo.py`: the modulator inputs in replay; sign enforcement after each step.
- `brain/arch/_brain.py`, `agent/agent.py`: `food_ingested`, `max_food_ingested`.
- `connectome/neuromodulators.py`, `data/neuromodulators/`: the circuit and its loader.
- `scripts/campaigns/generate_lawn_configs.py`: the modulated arms, random-sign and sign-grounded.
- `scripts/analysis/lawn_states.py`, `scripts/analysis/body_kinematics_eval.py`: acute arms, the speed
  measure, the gates and the knockout readings with Holm.
- Tests, Logbook 090, tracker B.3, roadmap.
