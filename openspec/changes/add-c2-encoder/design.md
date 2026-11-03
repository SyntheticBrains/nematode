## Context

- **The connectome brain today** (`brain/arch/connectome_ppo.py`; batched as `BatchedConnectome`). Its free values:

  - about 3,709 chemical weights, drawn at random. Even `count_scaled` multiplies counts by a random |z| and sign, and edges without a grounded sign keep a random sign;
  - 18 per-neuron food gains onto ASE, AWC and AWA;
  - a 5×4 pooled readout with no bias;
  - 5 action log-SDs;
  - the data layer's 104 signed and 17 fixed-sign strengths.

  Timescales are config values (60 s by default). Food contact and satiety never reach the network. The only slow state is the 55 ligand pools.

- **What reads what:** `BatchedConnectome` reads the brain's tensors live on each step (`food_gains`, `readout`, `log_std`, `w_chem`, `g_gap`, the modulator strengths and `_modulator_decay`). Writing a vector in place therefore takes effect without rebuilding. Knockout maps and the build-time weight scales are fixed when the brain is built.

- **The data:**

  - Emmons 2024 (the configs' source) carries EM serial-section synapse counts;
  - `synapse_signs: per_connection` grounds 3,176 edges (2,634 positive, 542 negative);
  - the 533 others come from presynaptic neurons releasing dopamine, serotonin or octopamine, or with no known transmitter;
  - the data layer tags each ligand `monoamine` or `peptide` (4 and 51).

- **Cost** (Logbook 104, and a timing this planning made on discarded seeds): about 6.5 µs per animal-step on one core at n = 200. C.2 is fitted on D_fit (the small lawn, the uniform well and kinematics), which needs two assays: about 1.44 M animal-steps, roughly 10–13 s per 200-animal evaluation.

- **Lessons carried from C.1** (Logbook 121):

  - check a model against what the benchmark grades before registering a fit;
  - tighten the plateau's guard, since a median near 507 stopped a restart;
  - keep seed blocks at least 2 M animals apart.

## Goals / Non-Goals

**Goals:**

- C.2's encoder, about 200 named parameters with a unit box, in both the per-step runner and the batched evaluator;
- every existing path unchanged;
- the evidence F.0b-ii needs before its registration: cost, slow-state feasibility, and a readiness probe whose answers can be its truths.

**Non-Goals:**

- registering F.0b-ii or C.2, and fitting C.2;
- W's nulls. Each null's construction must keep the encoder well defined, which this design states, but the nulls are W's;
- per-synapse or per-class-pair synaptic fitting.

## Decisions

01. **The switch:** `encoder: compact` on `ConnectomePPOBrainConfig`, default `none`.

    - **Unset,** no tensor, draw or output changes.

    - **Set,** the brain refuses unless all of these hold:

      - `dynamics: settling`;
      - `sensing_mode: klinotaxis`;
      - `modulators: true` with `modulator_layer: data`;
      - `modulator_satiety_link: true`;
      - `synapse_signs: per_connection`;
      - `weight_prior` unset;
      - no predator or thermotaxis projection;
      - the configured step and `modulator_step_seconds` equal.

      That last check guards the 5-s default.

02. **The fixed wiring.**

    - **Chemical weights:**
      - an edge i → j with grounded sign s has weight θ_s · s · c_ij / √(Σ_k c_kj²), where c is its synapse count and the sum runs over j's inputs. This is `count_scaled`'s normalisation without the random draw;
      - θ_exc scales the positive edges and θ_inh the negative ones;
      - an unsigned edge from a presynaptic neuron of transmitter group g has weight γ_g · c_ij / √(Σ_k c_kj²), with g ∈ {dopamine, serotonin, octopamine, none} (`atlas` transmitters).
      - The four groups cover every unsigned edge (checked in planning, `read_atlas_transmitters`): no known transmitter 317 edges from 20 neurons, dopamine 163 from 8, octopamine 42 from 2, serotonin 11 from 2.
    - **Gap junctions:** the data weights (the configured mode and normalisation) × θ_gap. Derived inverses are rebuilt when θ_gap is written.
    - **Determinism:** the weights depend on the data and the vector alone, not the brain's seed.

03. **Sensory inputs,** replacing `food_gains` when the encoder is on.

    - **The classes:** the 11 amphid chemosensory classes AWC, AWA, ASE, AWB, ASK, ADF, ASI, ASJ, ASG, ADL and ASH.
    - **Their features:**
      - concentration and dC/dt (tanh, as today), each with a gain per class, shared by the class's left and right neurons;
      - lateral (right minus left), with an antisymmetric gain per class: the right neuron gets +g·lateral and the left −g·lateral. That is 33 gains.
    - **Food contact** (the lawn density under the worm, in [0, 1]) goes into CEP (4 neurons), ADE and PDE with one gain each, the dopaminergic mechanosensors of bacteria behind basal slowing (Sawin et al. 2000). That is 3 gains.
    - **Satiety** enters through the existing satiety link on the ingestion drive onto NSM, which adds no parameter. Within an assay it barely moves, since it is a 12-h reserve.
    - **The per-step runner** feeds food contact from the same sense the `food_contact` module reports.

04. **The readout:** the 5×4 pooled readout (VB, DB, VA and DA means) plus 5 biases, so logits = pooled · readoutᵀ + bias. The 5 log-SDs are fitted.

05. **Neuromodulation:**

    - **The strengths:** the 104 signed strengths are written directly, and the 17 fixed-sign ones as strengths, with ρ = softplus⁻¹(s).
    - **The timescales:** τ_mono for dopamine, octopamine, serotonin and tyramine, and τ_pep for the 51 peptides. Decay is exp(−step / τ).

06. **The unit box** (`connectome/encoder.py`). Its centre is C.2's x₀, set without search data (D38).

    | Block | Range | Centre |
    |---|---|---|
    | θ_exc, θ_inh, θ_gap | [0, 3] | 1.5 |
    | γ (4) | [−2, 2] | 0 |
    | sensory gains (36) | [−5, 5] | 0 |
    | readout (20), bias (5) | [−3, 3] | 0 |
    | log-SD (5) | [−4, 1] | −1.5 |
    | signed strengths (104) | [−3, 3] | 0 |
    | fixed-sign strengths (17) | [0.001, 3] | 1.5 |
    | τ_mono, τ_pep | log [10, 3000] s | 173 s |

    The ranges are provisional. The readiness probe reports every parameter its answer leaves at a bound, and F.0b-ii's registration fixes the box.

07. **The vector interface:**

    - `encoder_parameters()` gives the 196 names in a fixed order;
    - `encoder_box()` gives each name's range and kind;
    - `write_encoder(brain, z)` writes a unit-box vector in place, under `no_grad`;
    - `read_encoder(brain)` returns it.

    A write followed by a read round-trips exactly. Under PPO, the brain's learnable parameters are the encoder's tensors and nothing else. That composition is differentiable, so C.3 trains the same architecture.

08. **The mutant encodings,** which bind the registered table's C.2 row:

    - **tph-1:** `modulator_knockout_genes: [tph-1]`;
    - **pdfr-1:** `[pdfr-1]`;
    - **tax-4:** the 21 sensory gains of the 7 tax-4 classes among the encoded inputs (AWC, AWB, ASE, ASG, ASI, ASJ, ASK) are zeroed. AFD, URX, AQR, PQR and BAG receive no encoded input, so nothing changes for them. AWA, ADF, ADL and ASH keep theirs.

    A mutant brain is built with its knockout genes, and the wild type's vector is written into it.

09. **Parity and identity:**

    - **Parity:** the encoded brain meets the batched evaluator's layered parity with the per-step runner (state within 1e-5 over 50 steps; bouts and trajectories as registered), on seeds 5001–5032.
    - **Identity:** a baseline is recorded before any edit on the hand-placed and data-layer connectome arms (batched runs on discarded seeds, all four v3 assays). Afterwards every record must be byte-identical with the encoder unset. The existing parity tests also pass unchanged.

10. **Configs:**

    - data-layer primitives assay configs for v3's four assays, carrying `encoder: compact`, `synapse_signs: per_connection`, `modulator_layer: data` and `modulator_satiety_link: true`;
    - `generate_assay_configs.py` gains a `datalayer` arm;
    - `benchmark_grade.ARMS` gains `connectome_c2`.

11. **The probes** (description, discarded seeds; `scripts/analysis/c2_encoder_probe.py`):

    - **Cost pilot:** one generation's pace, 16 simulations on 16 workers, for D_fit's two assays and for all four. It projects F.0b-ii, C.2 and W.
    - **Feasibility of a ~9-minute decay** (the roadmap's risk row):
      - a hand-set vector (τ_pep 540 s, one peptide coupling from a producer that is active on food onto the motor pools) runs López-Cruz's protocol;
      - the post-removal onset rate's fitted decay constant must lie within 6–12 min;
      - it reads no López-Cruz data, since local search is held out for C.2.
    - **Readiness probe:**
      - one sep-CMA restart from the box centre, σ₀ 0.2, population 16;
      - the ranking score against v3's band centres on D_fit's families;
      - the plateau rule once σ ≤ σ₀ and **every median in the last 40 generations is below 100**, so that more than half of each generation's candidates leave no statistic undefined (the guard Logbook 121 asked for);
      - a cap of 1,000 generations;
      - its answer is graded with presence on 5 populations on all five families;
      - reported: which families pass, the parameters at a bound, the reversal share, and the slow state's fitted timescales;
      - ready means the answer passes D_fit's three families in at least 3 of 5. Local search and chemotaxis are held-out predictions, reported, not required.

12. **Seeds** (all discarded):

    | Seeds | Use |
    |---|---|
    | 61,000,000 + 200·k + i, k = 16·(n − 1) + m (to 64,199,999); CMA 5980 | the readiness probe's fit |
    | 65,000,000 + 200·g + i | its grading |
    | 65,500,000 | the cost pilot |
    | 65,800,000 | the feasibility check |
    | 5001–5032 | parity, as registered |

## Risks / Trade-offs

- **[196 parameters is a slow fit]** → The cost pilot projects it, and the readiness probe measures where its plateau falls. If W would exceed its budget, fewer draws per arm is the roadmap's stated response.
- **[The encoder favours the wild type]** → Class-tied gains and fixed data weights are defined on neuron identities. W's nulls rewire edges and keep identities, so the encoder applies to them unchanged. The plan review's class-preserving null is W's.
- **[The amphid class list and food-contact targets are a choice]** → They are stated with their sources, and tax-4's zeroing follows from the registered table.
- **[Antisymmetric lateral assumes a head-sweep comparison]** → That is the klinotaxis sense the brain already uses: one scalar, right minus left.
- **[Food contact adds an input C.3's per-step runner must supply]** → The runner already reports it, and parity covers it.
- **[The feasibility check could fail]** → Then the claim is scoped to the faster assays, as the roadmap's risk row states.
