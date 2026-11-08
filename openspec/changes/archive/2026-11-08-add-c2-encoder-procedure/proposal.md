## Why

Logbooks 126–129 built a fitting procedure for C.2 in probe scripts, by monkeypatching the encoder's forward and bounding its vector, with no package code changed. With it, C.2 fitted to the wild type and Scheer's three mutants passes the wild type in 3 of 5 fresh populations with presence, and tph-1, pdfr-1 and tax-4 in 5 of 5 (Logbook 129). A registered fit, and every W arm, needs that procedure in the package: written down, tested, byte-identical when off, and the same in both runners.

The maintainer's decisions (2026-11-08):

- **Inhibition:** the inhibitory scale floored at 0.09, the value the passing fit used, recorded as chosen from the probes.
- **Receptor signs:** PDFR-1 (Gαs) and SER-5 (Gαq) switched on. The other 22 that a literature pass resolved go into the cited table, switched off.
- **The start:** a clean, staged start from the box centre, the same for every arm, with a readiness pilot first. That is Change B's, not this change's.
- **The held-out test:** Scheer's held-out cohorts in every fit, plus one leave-one-out holding out tax-4. Also Change B's.

This change is the encoder half, engineering only. The fit script, the D37 amendment and the registration follow in Change B.

## What Changes

- **`encoder_readout: linear | soft_cap`** (default `linear`). `soft_cap` passes each output's pre-squash value through 2·tanh(x/2).
- **`encoder_pool_centres: false | true`** (default `false`). The readout becomes b + Σ_j w_j (pool_j − c_j), with four fitted centres, one per pooled motor class, on [−1, 1].
- **`encoder_box: wide | graded`** (default `wide`). `graded` holds the three wiring scales on [0, 0.75] and every sensory gain on [−2, 2]. Every other range is unchanged.
- **`encoder_inhibition_floor: float | None`** (default `None`). It raises the inhibitory wiring scale's lower bound, which must lie inside the box's wiring range.
- **`modulator_receptor_signs: list[str]`** (default empty). Each named receptor's couples take the sign of its cited G-protein coupling from `data/neuromodulators/receptor_couplings.json`, under `receptor_signs.json`'s rule: Gαi/o and Cl⁻ channels −1, Gαq and Gαs +1.
  - A receptor not in the table, or with a null sign there, is refused at load.
  - Under per-ligand tying, a couple signed this way is a fixed-sign group of its own, one per ligand and receptor. Its magnitude is not shared with the ligand's other fixed couples.
- **`data/neuromodulators/receptor_couplings.json`:** the 54 receptors behind the layer's fitted couples. 24 resolve to a sign; 30 stay null. Every signed entry carries a citation and a quote checked against its source. `receptor_signs.json` and `layer.json` are unchanged.
- **The vector's order:** the base entries, then the speed noise's timescale (with `encoder_speed_noise: ou`), then the four centres. Any stored vector stays a prefix.
- **Both runners share the encoded forward,** so the per-step runner and `BatchedConnectome` compute the same.
- **An optional forward observer** on `ConnectomeTopology`, `None` by default: called after each encoded settling with the outputs and hidden state, so a fit's guards need not replace the forward.
- **Under PPO,** the centres are learnable and the cap is differentiable. The box and the floor bound only the unit vector that CMA writes.
- **Checks:**
  - byte-identity of every existing connectome output with the options off (`connectome_identity_check.py`);
  - parity between the runners with every option on;
  - unit tests;
  - a reproduction check: the package's forward computes what the probe's patched forward computed on the passing answer, to 1e-5 and step for step; its five-population grade is re-run as description.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `connectome-encoder`: the soft-capped readout, the pool centres, the graded box, the inhibition floor, the vector's order and counts (a modified requirement), the forward observer and the probe answer's reproduction.
- `neuromodulator-layer`: cited receptor signs behind an empty default.
- `batched-evaluation`: parity with the per-step runner with every new option on.

## Impact

- `quantumnematode/brain/arch/connectome_ppo.py`: the fields, their refusals, and the cap and centres in `_forward_encoded`.
- `quantumnematode/connectome/encoder.py`: the graded ranges, the floor, the centre entries, the table-signed groups, and their write and read.
- `quantumnematode/connectome/neuromodulator_layer.py`: `load_data_layer(..., receptor_signs=)`.
- `data/neuromodulators/receptor_couplings.json` and its `PROVENANCE.md` entry.
- `utils/step_duration.py`: the new fields' classification.
- `scripts/analysis/c2_encoder_probe.py`: a converter from the probe layout to the package's.
- Tests, AGENTS.md, CHANGELOG and the tracker.
