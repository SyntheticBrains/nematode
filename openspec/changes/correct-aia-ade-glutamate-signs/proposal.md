## Why

The per-connection sign table signs all 29 glutamatergic chemical edges onto AIA and ADE as excitatory. None has a physiology override, and Fenyves et al.'s expression step gives none of them a sign, so each falls to the per-neuron rule, which signs glutamate positive.

López-Cruz et al. 2019 (*Neuron* 102:407–419) show the opposite for the circuit that drives local search: "Glutamatergic chemosensory and mechanosensory neurons generate local search by suppressing MGL-1-expressing neurons". The chemosensory neurons are "a set of glutamatergic chemosensory neurons that synapse onto the AIA interneurons", and the mechanosensory ones "a set of glutamatergic mechanosensory neurons that synapse onto the ADE sensory-inter neurons". MGL-1 "appears most similar to mammalian Gi/o-coupled Group II metabotropic receptors".

C.2's wild type, which fixes its wiring from this table, is therefore wired with the opposite sign on the pathways its held-out local search runs through. The planned perturbation family grades López-Cruz's cell-level rows, and it would grade them against that contradiction.

Nothing registered reads C.2 yet. The readiness restarts, F.0b-ii and W come next, so this is the point at which to correct the wild type.

The maintainer's decisions (2026-11-04): correct the sign now, as a cited physiology override; do not model MGL-1 as a slow modulator; grade the perturbation family separately from benchmark v3.

## What Changes

- **16 physiology overrides citing López-Cruz 2019** in `data/connectome/sign_overrides_physiology.csv`, each signing an edge inhibitory. They cover the edges whose presynaptic cells the paper's manipulations covered:
  - glutamatergic tax-4 chemosensory cells onto AIA: ASE, ASG, ASK and AWC, 12 edges;
  - glutamatergic mec-3 mechanosensory cells onto ADE: FLP and AVM, 4 edges.
- **The other 13 glutamatergic edges stay on the rule.** These are ASH, ADL, AIZ, AIM, ADA and AIB onto AIA, and OLL and IL1 onto ADE. None of the paper's cell-specific manipulations covered these cells (only its pan-neuronal knockout did), and the table's physiology step requires a measurement of the pathway. The design records this as a limit.
- **`lopezcruz2019` joins `PHYSIOLOGY_CITATIONS`.**
- **`sign_override_exclude`** on `ConnectomePPOBrainConfig`: a list of citation keys whose overrides the brain ignores, empty by default. `[lopezcruz2019]` reproduces the table as it stood before this change.
  - The nine committed per-connection configs behind recorded results set it, so they keep reproducing their records: five lawn configs (`*_lawns_modulated_perconn*`, written by `generate_lawn_configs.py`) and four thermal target-35 configs (written by `generate_sign_rung_configs.py`). Both generators write it, and `sign_scale_calibration.py`, which builds per-connection brains in code for a recorded calibration, passes it.
  - The seven encoded configs take the corrected table.
- **The table's recorded composition changes:** physiology 51 → 67, the rule 1,449 → 1,433. The Wormlight comparison runs on the table with `lopezcruz2019` excluded, and the full table differs from that on exactly the 16 edges.
- **Checks:**
  - `connectome_identity_check.py` gains a brain-digest job for the nine excluded configs and one encoded config, recorded before the change;
  - every connectome output that does not read the table is byte-identical;
  - the nine excluded configs build brains identical to before;
  - an encoded brain's fixed normalised counts change in exactly the 16 entries, each moving from the excitatory matrix to the inhibitory one at the same magnitude.
- **An addendum to Logbook 122**, and a dated note on Logbook 091's counts.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `connectome-substrate`: the per-connection sign table gains the López-Cruz overrides and its recorded composition changes.
- `connectome-ppo-brain`: a brain can exclude overrides by citation, to reproduce an earlier table.

## Impact

- `data/connectome/sign_overrides_physiology.csv`, `data/connectome/PROVENANCE.md`
- `quantumnematode/connectome/signs.py`: the citation key, and an exclusion parameter on `per_connection_signs`
- `quantumnematode/brain/arch/connectome_ppo.py`: the config field and its pass-through
- `scripts/campaigns/generate_sign_rung_configs.py`, `scripts/campaigns/generate_lawn_configs.py`, `scripts/analysis/sign_scale_calibration.py` and the nine configs
- `scripts/analysis/connectome_identity_check.py`: a brain-digest job
- the tests in `tests/quantumnematode_tests/connectome/test_signs.py` and `tests/quantumnematode_tests/brain/arch/test_connectome_per_connection_signs.py`, with their pinned counts
- C.2's readiness records in Logbook 122 stay as recorded, on the old wiring; the readiness restarts read the corrected one
