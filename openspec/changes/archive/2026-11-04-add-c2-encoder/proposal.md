## Why

C.2, the connectome controller, is fitted by sep-CMA on the benchmark's distance (D38). F.0b-ii must first show that sep-CMA can recover it. Neither can start, because **C.2 has no encoder**: no defined set of fitted parameters. The roadmap lists "a connectome encoder" among what the repository lacks, the tracker's design still has "C.2: the encoder's parameterisation" as an open question, and the registered mutant table's tax-4 row "binds whatever encoder F.0b-ii builds".

Today's connectome brain has about 3,873 free values, nearly all of them 3,709 randomly initialised per-synapse weights. sep-CMA's generations grow roughly with the parameter count, so fitting every synapse would make W's equal-budget nulls infeasible on this machine.

The maintainer's decisions (2026-11-03):

- **a compact encoder of about 200 parameters:**
  - the wiring fixed from data;
  - per-class sensory gains, with food contact added;
  - the motor readout with biases;
  - the neuromodulator layer's 121 strengths and two tied timescales;
- **the 533 chemical edges without a grounded sign** keep one signed gain per presynaptic transmitter group;
- **the encoder lives in the brain, in both runners,** so C.3 trains the same architecture by reward;
- **build, then probe:** a cost pilot and a C.2 readiness probe on discarded seeds follow, before F.0b-ii's change. The probes' answers become its truths, as C.1's did;
- **at the spec review, because W is the phase's longest run:**
  - fix W's inheritance here: the gap model (strength-normalised, the corrected model of Logbook 103, for every arm), how a null is encoded, the compute path, the cost per fit, and the fit-to-fit spread that sizes W's draws per arm;
  - build a modulator-tying switch, so F.0b-ii can choose 196 or 132 parameters on that evidence.

## What Changes

- **`encoder: compact`** on `ConnectomePPOBrain` (default none, byte-identical when unset). It turns the brain into a function of 196 named parameters (132 under `per_ligand` tying):

  | Block | Parameters | Count |
  |---|---|---|
  | wiring | excitatory, inhibitory and gap scales on fixed data weights | 3 |
  | unsigned edges | a signed gain per presynaptic transmitter group (dopamine, serotonin, octopamine, none) | 4 |
  | sensory | class-tied gains: concentration, dC/dt and lateral (antisymmetric) into 11 amphid chemosensory classes; food contact into CEP, ADE and PDE | 36 |
  | readout | the 5×4 pooled motor readout and 5 biases | 25 |
  | noise | the 5 action log-SDs | 5 |
  | neuromodulation | 104 signed and 17 fixed-sign strengths | 121 |
  | slow state | the monoamine and neuropeptide timescales | 2 |

- **Its wiring is fixed from data:** each chemical edge's weight is its grounded sign × its synapse count, normalised per postsynaptic neuron. Gap junctions keep their data weights. Satiety enters through the existing satiety link on the ingestion drive.

- **A unit-box interface:** `encoder_parameters()`, `encoder_box()`, `write_encoder(z)` and `read_encoder()`, so sep-CMA writes a vector and the batched evaluator reads it live. The same parameters are the brain's learnable parameters under PPO.

- **The tax-4 encoding:** zeroing the gains of the tax-4 classes among the encoded inputs (AWC, AWB, ASE, ASG, ASI, ASJ, ASK). tph-1 and pdfr-1 keep their gene knockouts.

- **Configs:** data-layer primitives assay configs with the encoder, for v3's four assays, and a `connectome_c2` arm in `benchmark_grade`.

- **Checks:**

  - parity between the per-step runner and the batched evaluator for the encoded brain;
  - byte-identity for every existing connectome path;
  - a cost pilot;
  - a feasibility check that the slow state can carry a ~9-minute decay;
  - the encoder on every existing null wiring, with parity on one null;
  - a readiness probe: up to 3 sep-CMA restarts against v3's D_fit on discarded seeds, graded with presence. It measures the restart-to-restart spread of held-out predictions and mutant differences against W's minimum;
  - a profiled cost pilot, with a CPU-against-MPS benchmark of a whole generation's settling step.

- **Logbook 122.**

## Capabilities

### New Capabilities

- `connectome-encoder`: C.2's compact parameterisation of the connectome brain, its unit box, its mutant encodings and its checks.

### Modified Capabilities

- `batched-evaluation`: the encoded brain under the batched evaluator, with parity and byte-identity when the encoder is off.

## Impact

- **Code:**
  - `quantumnematode/brain/arch/connectome_ppo.py` (the encoder's wiring, inputs, readout bias and timescales);
  - `quantumnematode/connectome/encoder.py` (new: the parameter table, box and vector interface);
  - `quantumnematode/evaluation/controllers.py` (`BatchedConnectome` reading the encoded tensors and food contact);
  - `scripts/campaigns/generate_assay_configs.py` (a data-layer arm);
  - `scripts/analysis/benchmark_grade.py` (`ARMS`);
  - `scripts/analysis/c2_encoder_probe.py` (new: the cost pilot, the feasibility check and the readiness probe).
- **Records:** `docs/experiments/logbooks/supporting/122-c2-encoder/`; Logbook 122; the tracker (F.0b-ii, C.2, design.md's open question); AGENTS.md; the CHANGELOG.
- **Cost:**
  - engineering: several days;
  - the probes: the cost pilot first; then up to 3 readiness restarts at a cap of 1,000 generations, each projected by the pilot;
  - nothing registered; F.0b-ii and C.2 follow in their own changes.
