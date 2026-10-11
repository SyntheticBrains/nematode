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
  - **Food:** NSM receives a sensory current proportional to the food the worm ate on its last move.
  - **Replay:** the concentrations are carried with the buffered states, so PPO's replay sees them.
- **Knockouts.** A config switch removes serotonin (*tph-1*), PDF (*pdf*), MOD-1 (*mod-1*) or PDFR-1
  (*pdfr-1*). It is applied at evaluation for the primary reading, and from the start of training for
  the raised mutants beside it.
- **Food ingestion reaches the brain:** `BrainParams.food_ingested`.
- **A registered panel:**
  - **Gates:** the modulated connectome learns and still dwells on D.1's cell.
  - **The primary reading:** its acute *tph-1* and *pdfr-1* knockouts, graded on the published
    directions (*tph-1* roams more, *pdfr-1* dwells more).
  - **Beside it:** raised mutants and a satiety link.

## Capabilities

**Modified**:

- `connectome-ppo-brain`: the modulator field, its knockouts, and its persistence across steps and
  in replay.
- `brain-architecture`: `BrainParams.food_ingested`.

## Impact

- `brain/arch/connectome_ppo.py`: the field, gating, release, ingestion drive, knockouts and replay.
- `brain/arch/_brain.py`, `agent/agent.py`: `food_ingested`.
- `data/neuromodulators/`: the circuit.
- `scripts/campaigns/generate_lawn_configs.py`, `scripts/analysis/lawn_states.py`: the modulated arms
  and the knockout readings.
- Tests, Logbook 090, tracker B.3, roadmap.
