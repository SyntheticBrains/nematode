## Context

On log-compressed gap counts the corrected model shows no wiring effect on either cell. That is the
registered STOP condition (Logbooks 099 and 100). On 2026-10-20 the maintainer chose to open the
modelling question inside Phase 9: test other defensible gap models before the STOP statement ships.

The maintainer's choices:

- **models:** strength-normalised coupling and ohmic coupling at ×0.03;
- **cells:** the point worm first, and the body only for a model that reads `placement`.

**The artefact's mechanism** (code comment at `connectome_ppo.py`, gap-junction construction):

- `G[i, j] = w_ij / √(d_i d_j)`, where `d` is the binary gap degree;
- the normaliser counts partners, not synapses;
- so a high-count pair between low-degree neurons couples far above 1: ALA–CAN at about 90 under counts.

The two corrections tested so far change `w` (unit weights, log counts) and leave the normaliser as it
is. Ohmic coupling changes how gap current enters the settling map.

## Goals / Non-Goals

**Goals**

- Read the wiring contrast under two principled gap models on the point worm, at the program's minimum,
  under a verdict map that says what each outcome does to the STOP statement.

**Non-goals**

- The body contrast. A `placement` model gets its own registration.
- Count-capped weights, which the maintainer did not choose.
- Re-reading any committed panel.
- Choosing the "right" gap model. This change reads which models carry a wiring effect, not which one is
  true.
- Mechanism readings (G.1–G.3b).

## Decisions

### Decision A: Strength normalisation, as a brain option

The option is `gap_normalisation: Literal["degree", "strength"]`, defaulting to `degree`.

Under `strength`, the matrix of pair weights `w` (after `gap_weight_mode`) is divided symmetrically by
`√(s_i s_j)`, where `s_i = Σ_j w_ij` is neuron `i`'s gap strength. It is computed on each wiring's own
matrix, as the degree normaliser is now. `gap_weight_scale` still multiplies afterwards.

**What this means for the null.** A degree-preserving swap keeps every neuron's partner count, so under
`degree` both wirings share each neuron's normaliser. It does not keep strengths: the counts travel with
their edges, so under `strength` the gap-only null re-normalises each neuron by its own new strength.
The null therefore differs from the wild type in placement, in each neuron's total gap weight (as under
`degree`), and now also in each neuron's normaliser. D24's table reports the per-neuron strengths beside
the totals (Decision E).

A neuron with no gap partners has zero strength and keeps zero couplings.

Properties, each pinned by a test:

- `G[i, j] ≤ 1`, since `w_ij ≤ s_i` and `w_ij ≤ s_j`;
- `G` is symmetric with spectral radius 1 (the symmetric normalisation of a non-negative symmetric
  matrix), so the gap path adds at most unit gain per settling step;
- with `gap_weight_mode: unit`, `s = d`, so `strength` equals `degree` exactly: a built-in identity
  check;
- `degree`, the default, is byte-identical to today (pinned digests, state-dict keys, outputs, the next
  draw);
- a neuron with zero strength has an all-zero row and column.

It is refused:

- with gap junctions off, where there is nothing to normalise;
- with `plastic_gaps`, which learns the scaled matrix directly, so the bound would not be maintained.

It composes with `gap_coupling: ohmic`, but that combination is not read here.

### Decision B: The ohmic model, unchanged

The ohmic model is `gap_coupling: ohmic` at `gap_weight_scale: 0.03`, with additive weighting by
counts, on G.0-A's four `*_g0a_ohmic_*_t35` configs, unchanged. Its scale was calibrated on untrained
brains before any run (Logbook 097), and is not revisited.

Its wiring gap was barred from reading in G.0-A's pilot. **That pilot's gap stays unread.** This panel
reads ohmic on fresh seeds.

### Decision C: The probe for strength, before any run

The probe uses 097's instrument: `sign_scale_calibration.measure` on the 64 fixed inputs
`sign_scale_calibration.inputs()`, with each arm's brain config from `gap_substrate_selection._connectome_config`,
on the strength pilot's seeds. It is implemented in the new harness's `probe` subcommand, because
`gap_substrate_selection.probe` is hard-wired to G.0-A's grid. It measures the strength wild type's
action spread and saturated fraction against the gap-off brain, and against the count and log models
beside.

**The rule** is fixed now, and is the one the ohmic scale had to meet: the strength model's spread must
be at least half the gap-off brain's. If it is not, the model does not run, and the next step is a dated
decision. No scale is calibrated for strength: its couplings are bounded by construction, which is the
point of the model.

### Decision D: The panel and the readings

**The panel.** The thermal cell at target 35 on Emmons 2024, at block V's point, for 3,000 episodes:

- for each model, the wild type and the gap-only null, each learning and frozen;
- seeds 4201–4264, the same 64 for both models;
- **512 runs**.

**The readings.** `gap(strength)` and `gap(ohmic)` on `auc_success`, wild type − null, each paired by
seed:

- an 80% bootstrap interval for each;
- **BH-FDR across the family of two** (`ops.apply_family_correction`), on the two-sided Wilcoxon p;
- each classified by `mc.classify` at **m = 0.0577**.

**Gates**, per model: each learning arm beats its wiring's frozen floor on all 64 seeds, and the two
learning arms do not both plateau at 90% or more (`mp.level_passes`).

| state, per model | verdict |
|---|---|
| gates fail, or the probe stopped the model | `unreadable` |
| `move_wt` | **`placement`** |
| `below`, either direction | `below_minimum`, with the direction recorded |
| `no_move` | **`no_wiring_effect`** |
| `move_null` | `reversed` |
| `unresolved` | `unresolved` |

**What the pair of verdicts does to the STOP statement**, fixed now:

- **Either model reads `placement`:** the STOP statement is qualified. The wiring effect depends on how
  electrical synapses are modelled: present under that model, absent under unit weights and log counts.
  Each `placement` model registers its body contrast next. Whether G.1–G.3b reopen on it is a dated
  decision after the body reads.
- **Both read `no_wiring_effect`:** the STOP statement stands, with four corrections behind it (unit
  weights, log counts, strength normalisation and ohmic coupling). Ohmic's reading is stated at its
  operating point: at ×0.03 the median neuron keeps 93% of its own drive, and the untrained action
  spread is 0.0162 against the gap-off brain's 0.025 (Logbook 097). An ohmic null is therefore a null at
  weak coupling, and is never read as a null at every conductance. The 9a synthesis follows.
- **Any other pair:** a dated decision.

**Described beside, never verdicts:**

- episodes to 30% success;
- each arm's plateau and floor;
- the two wild types against each other.

### Decision E: D24 and the gates, before launch

**D24.** `gap_operating_point.matched_statistics` is computed for both models' wild types and nulls on
the 64 panel draws, and committed as `table.json` before launch. For ohmic it reads the built
conductance matrix at ×0.03. For strength it also reports each wiring's per-neuron strengths (ALA, CANL,
AVAL), the normalisers the null changes. The shortcut audit (sensor–motor gap pairs, motors at one and two hops)
is the same as for every gap-only null.

**Gates.** The preflight evaluates each model's level:

- ohmic, on G.0-A's pilot arms (9871–9886), where it read eligible: 61.1% and 54.1%, floors 0%;
- strength, on its own pilot (9905–9912, all four arms, configured as the panel).

A level that fails its floor, saturates, sits near the bar, or has no evidence blocks the launch.

**Cost** is taken from the pilots' own run records. In G.0-A's pilot, ohmic runs took as long as
log-count runs: a median of 23.2 minutes learning and 4.8 frozen, against 22.5 and 4.5.

### Decision F: Sizing and the operating characteristic

The spread is taken as σ = 0.119, G.0's count level over 64 seeds and the largest committed on this
cell. At 64 seeds a single model's characteristic is Logbook 099's: a true null reads
`no_wiring_effect` about 80% of the time, and a lead of 0.08 reads `placement` about 94% of the time.

The family-of-two correction raises q slightly. The operating characteristic is re-simulated with the
registered instrument, including the correction, and stated in the launch record.

## Risks / Trade-offs

- **Strength normalisation is new.** The probe protects against an unlearnable operating point, and the
  gates against a model that does not learn.
- **Ohmic at ×0.03 couples weakly.** A null there may mean the coupling is too weak to carry placement.
  The reading is stated at its operating point, never generalised to every conductance.
- **Ohmic learns more slowly.** Its plateau is lower (61% against 70%), so its contrast reads a slower
  learner. That is a property of the model, and it is stated beside the reading.
- **Two models raise the chance of a spurious `placement`.** The BH-FDR family of two corrects for it.
  The point worm's `placement`, if it appears, is then tested again through the body before anything
  reopens.
- **This defers the STOP statement** by about two days of work and about 9 machine-hours.
