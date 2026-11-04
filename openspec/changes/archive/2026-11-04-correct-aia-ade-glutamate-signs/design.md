## Context

`quantumnematode.connectome.signs.per_connection_signs` signs each chemical edge by the first of four steps: a cited physiology override, Fenyves et al.'s expression polarity, the per-neuron transmitter rule, or no sign. Its 51 overrides quote a measurement of each pathway, for example "The AWC neuron inhibits AIY interneurons through glutamate-gated chloride channels".

Every glutamatergic edge onto AIA and ADE reaches the rule and is signed +1. C.2's encoder fixes its wiring from this table (`synapse_signs: per_connection`, grounded sign × count, normalised per postsynaptic neuron). A rewired null takes each moved edge's sign from its origin (`_per_connection_edge_signs`).

## Goals / Non-Goals

**Goals:**

- Sign the pathways López-Cruz 2019 measured inhibitory, before C.2's readiness restarts.
- Keep every committed record reproducible.

**Non-Goals:**

- Modelling MGL-1's kinetics, a slow modulator coupling, or the fast glutamate-gated channels on AIA.
- Any other edge, and any other sign source.

## Decisions

### D1. Override the 16 edges the paper's cell-specific manipulations covered, not all 29

The paper's chemosensory knockout uses the tax-4 promoter, and its glutamatergic tax-4 cells that synapse onto AIA are ASE, ASG, ASK and AWC: 12 edges with 195 synapse units between them. The mechanosensory knockout uses mec-3, and its glutamatergic cells onto ADE are FLP and AVM: 4 edges, 35 units.

Each edge's evidence is the paper's conclusion for its pathway, the same standard as the AWC → AIY override. Glutamate loss in one module alone barely changes local search, because the two modules are redundant. In the vendored rows, early reorientations are 5.7 for the tax-4 set and 6.3 for the mec-3 set against a control's 7.5; 6.2, 4.1 and 6.7 for AWC, ASK and ASG against 5.7, 5.2 and 7.2. Losing both modules takes them to 2.3, as mgl-1's loss does (2.0). mgl-1's rescue in AIA and ADE restores them, and silencing AIA or ADE in mgl-1 mutants restores them too. The paper's promoters for the two sets (tax-4 and mec-3) are verified from its methods before the rows are written.

The other 13 glutamatergic inputs come from cells no cell-specific experiment covered: ASH, ADL, AIZ, AIM, ADA and AIB onto AIA, and OLL and IL1 onto ADE. They could be signed from the receptor's location alone, since MGL-1 is postsynaptic in AIA and ADE. That would be an inference about the target, not a measurement of the pathway, which is the standard every other override meets. They stay on the rule, and this is recorded as a limit. ASH and ADL are large inputs onto AIA (24 and 19 units onto AIAL), so the limit matters, and the perturbation family's pan-neuronal rows will read it.

### D2. An inhibitory fast sign stands in for MGL-1

MGL-1 is metabotropic and, in the paper's words, "suppresses AIA and ADE neurotransmitter release" over an extended period. A rate network's fast sign captures the direction, not the timescale. The paper also says chemosensory glutamate "also activates fast glutamate-gated ion channels on AIA", without giving their sign.

The maintainer chose the override over a data-layer glutamate pool with an MGL-1 receptor: it's minimal, and it uses the table's existing mechanism. The perturbation family will show whether the direction alone carries the effects.

### D3. Exclude by citation, with the corrected table as the default

`per_connection_signs(..., exclude_citations=())` drops every override whose citation is listed, and `ConnectomePPOBrainConfig.sign_override_exclude: list[str] = []` passes it through. A citation key the table does not list is refused at config validation.

The corrected table is the default, following `continuous.reversal_kinematics`, whose new default came with an option to restore the old behaviour. Excluding by citation is general, so a later override set can be reproduced the same way, and it adds no second copy of the table.

### D4. Which configs keep the old table

The nine committed per-connection configs whose runs are recorded set `sign_override_exclude: [lopezcruz2019]`:

- `foraging/*_lawns_modulated_perconn{,_frozen,_pdfr1,_satiety,_tph1}.yml`;
- `thermal_foraging/*_per_connection_signs{,_frozen}_t35.yml` and their rewired-null pair.

Their generators write it too: `generate_lawn_configs.py` for the five lawn configs (its `modulated_perconn` variant) and `generate_sign_rung_configs.py` for the thermal four, so regeneration reproduces them. `scripts/analysis/sign_scale_calibration.py` builds per-connection brains in code for the calibration behind the thermal rung's sign scales; it passes the exclusion, so re-running it reproduces its record. The seven encoded configs take the corrected table: C.2 has no registered result.

### D6. The Wormlight anchor is kept on the excluded table

The comparison with Wormlight's export at `1190b3e` is a digest over the table's rows outside 23 known exceptions, taken from an export the repository does not vendor. With 16 more exceptions the digest cannot be recomputed from Wormlight's side. The table with `lopezcruz2019` excluded is the earlier table, so the comparison runs on it unchanged. A second check shows that the full table differs from the excluded one on exactly the 16 edges.

### D7. Verifying a change the identity check cannot see

`connectome_identity_check.py`'s jobs do not read the sign table, so their identity says nothing about this change. Before its baseline is recorded, it gains a brain job: a digest of the built brain's parameters, for each of the nine excluded configs and for one encoded config. After the change, the nine are identical and only the encoded job differs. An in-tree test shows the encoded brain's normalised counts differ from its excluded twin in exactly the 16 entries.

### D5. Nulls need no change

A null's edge takes its origin's sign, so a corrected edge moved by a rewiring carries −1 to its new place. A boundary-held null keeps all 16 edges in place.

## Risks / Trade-offs

- **C.2's encoder box was sized on the old wiring** (the wiring-scale range came from saturation at 0.5–1.5). Sixteen flipped edges out of 3,709 should move saturation little; the readiness restarts read the box on the new wiring anyway.
- **Logbook 122's readiness figures are on the old wiring.** The addendum says so, and they are not re-read as the new wiring's.
- **The 13 untested inputs** keep a sign the receptor's location argues against (D1).
- **Wormlight's export differs on 16 more edges.** The comparison test lists them, as it lists the 23 already.

## Migration Plan

Default-on for configs without the exclusion. Nothing to migrate: every committed config that should reproduce its records sets the exclusion in this change.
