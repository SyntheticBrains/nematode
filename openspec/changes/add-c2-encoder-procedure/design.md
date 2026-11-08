## Context

Logbooks 126–129 applied five pieces in probe scripts, by monkeypatching `ConnectomeTopology._forward_encoded` and bounding the CMA vector:

- the soft cap and pool centres (`staged_bc.py`);
- the graded box (`graded_v4.py`, `staged_bc.py`);
- the inhibition floor (`prototypes.py`);
- PDF's and ser-5's signs, as bounds (`prototypes.py`, `fit_mutants.py`);
- slow speed noise, already in the package (Logbook 123).

The passing answer is `docs/experiments/logbooks/supporting/129-c2-wild-type-and-mutants/probes/grade_nudged.json`.

## Goals / Non-Goals

**Goals:**

- every piece in the package, off by default and byte-identical when off;
- one forward for both runners;
- the passing probe answer reproduced in the package encoding.

**Non-goals:**

- the fit script, its guards, objective and stop rules;
- the D37 amendment, the start and the registration (Change B);
- C.3's PPO training;
- any change to `layer.json` or `receptor_signs.json`.

## Decisions

### D1. The soft cap is 2·tanh(x/2), inside the shared forward

`_forward_encoded` returns the mean that both runners and PPO read, so the cap sits there. It replaces nothing: `linear` returns the same tensor as before the option. The cap of 2 is the probe's (`staged_bc.SOFT_CAP`). It bounds each output's pre-squash value at ±2, where the action squash is still responsive.

### D2. Pool centres are four entries on [−1, 1], subtracted before the readout

The readout reads b + Σ_j w_j (pool_j − c_j). The probe's centres used c = −1 + 2u for a unit u, so the entries are linear on [−1, 1] with centre 0. At the box centre, then, the readout is the one without centres.

- **Under PPO** they are an `nn.Parameter`, among the policy's learnable parameters.
- **Their names** are `readout.centre.VB`, `.DB`, `.VA` and `.DA`.

### D3. The graded box changes only wiring and sensory ranges

- **`encoder_box: graded` sets** `_WIRING_RANGE` to (0, 0.75) for the excitatory, inhibitory and gap scales, and `_SENSORY_RANGE` to (−2, 2).
- **The unsigned-group gains, readout, noise, modulator and timescale ranges are unchanged.**
- **The box centre moves** to a wiring scale of 0.375 and a sensory gain of 0.

The probe's mapping (`staged_bc.to_encoder`) is the same map, so a probe unit vector is the package's unit vector for these entries.

### D4. The inhibition floor is the inhibitory scale's lower bound

`encoder_inhibition_floor: f` sets `wiring.inhibitory`'s range to (f, high). It must satisfy 0 ≤ f < high, or the config is refused.

- **The probe held the floor as a CMA bound** on the graded box's unit. The package holds it in the entry's range.
- **So a probe unit vector's inhibitory entry needs conversion,** u_pkg = (v − f)/(high − f), with v the value. The converter (D8) does it.

### D5. Receptor signs come from a cited table, applied at load

`load_data_layer(..., receptor_signs=("pdfr-1", "ser-5"))` sets the sign of every couple whose receptor is named, from `receptor_couplings.json`.

- **What is refused:** a receptor not in the table, with a null sign, or carrying no fitted couple in the layer.
- **The table:**
  - Schema: per receptor `coupling`, `sign`, `evidence`, `citation`, `quote`, `verified` and `note`.
  - `verified` records that the quote was checked against its source, with the date.
  - Only verified entries may be named.
- **Couples already fixed by `receptor_signs.json` keep their sign.** ser-5's couples are the layer's only fitted monoamine couples, and the table resolves them.
- **Under the receptor-placement null, a sign travels with its couple,** as fixed signs already do. The sign belongs to the receptor, not to the cells.

### D6. Under per-ligand tying, a table-signed couple is its own fixed group

The passing answer has ser-5's strength at 1.44 and serotonin's fixed group at 2.97. One shared magnitude could not represent it. So the groups are keyed:

- (ligand, `fixed`) for `receptor_signs.json`'s couples;
- (ligand, receptor) for the table's couples, named `modulator.fixed.<ligand>.<receptor>`;
- (ligand, `fitted`) for the rest.

The mechanism is a row field, `sign_source: str | None = None`. The loader sets it to the table's name on the rows it signs, and `_tied_name` returns `fixed.<ligand>.<receptor>` for such rows. The receptor-placement null builds its rows with `dataclasses.replace`, so the field travels with each moved couple and the null keeps the wild type's names.

A table-signed group's strength is the softplus of its value, log-scaled on the fixed-strength range (0.001, 3.0), as the fixed groups' are.

- **With PDFR-1 and SER-5 named:** pdf-1's and pdf-2's fitted groups become `modulator.fixed.pdf-1.pdfr-1` and `modulator.fixed.pdf-2.pdfr-1`, and serotonin's fitted group becomes `modulator.fixed.serotonin.ser-5`. The per-ligand count is unchanged at 132.
- **Under per-couple tying,** each named couple's entry moves from the signed range to the fixed-strength range.

### D7. The vector's order keeps every stored vector a prefix

The order is the base entries, then `timescale.speed_noise` (with `encoder_speed_noise: ou`), then the four centres. The probe stored its centres before the timescale, and the converter (D8) reorders them.

**The count:** the base (196, or 132 per ligand), plus 1 with speed noise, plus 4 with centres. The passing probe's configuration is 137 per ligand.

### D8. A converter and a reproduction check

`c2_encoder_probe.package_vector(probe_z)` maps a probe vector into the package encoding, entry by entry by name. The table-signed groups are renamed, for example `fitted.pdf-1` to `fixed.pdf-1.pdfr-1`, though their positions hold, since groups are ordered by first appearance:

- the graded entries unchanged;
- the inhibitory entry re-based on the floor;
- the PDF and ser-5 signed values re-expressed on the fixed log range, clipped at 0.001;
- the timescale and centres reordered.

**The reproduction check is a forward-level parity.** The converted answer in a package brain, and the probe's answer in the probe's patched brain, run the recorded states of one reference episode on the small lawn and one on food removal. Every output must agree within 1e-5, and every step's chosen primitive must be the same.

**The five-population grade is re-run as description,** reported beside the probe's, not as a gate. The probe's pass sits at the threshold: two populations failed by 0.03–0.22 z. The softplus round trip changes the PDF and ser-5 strengths at float32 precision, so a chaotic trajectory can diverge and move a population across the bar without anything being wrong.

### D9. Refusals

- **`encoder_readout`, `encoder_pool_centres`, `encoder_box` and `encoder_inhibition_floor` away from their defaults require `encoder: compact`.** The validator and the constructor both refuse otherwise, as the encoder's other fields are refused.
- **`modulator_receptor_signs` requires `modulator_layer: data`.**
- **The floor must lie inside its box's wiring range.**

### D10. An optional forward observer

`ConnectomeTopology.forward_observer: Callable[[Tensor, Tensor], None] | None = None`. When it is set, `_forward_encoded` calls it after settling with the outputs and the settled hidden state, detached; the return is unchanged. A fit's guards record the outputs' spread and the motor layer's saturation through it, where the probes replaced the forward. With no observer the forward is the same code path, byte-identical.

## Risks / Trade-offs

- **The floor and the cap are values chosen from probes, not from data.** They are recorded so in the field's docstring by their technical role, with no pointer to planning history, and in the logbook.
- **Table-signed groups add complexity to the tying.** The alternative, merging them into the ligand's fixed group, could not reproduce the passing answer.
- **The box and the floor do not bind PPO.** C.3 trains the raw tensors. That is stated, not enforced.
