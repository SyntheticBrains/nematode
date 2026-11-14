# Effort ledger

Every unit of work on the spike, so that Part B's estimates rest on measurement. Times are wall-clock.
"Agent" means work done by Claude or its subagents; "maintainer" means the maintainer's own reading
and decisions.

## Before session 0: designing the spike (2026-11-14)

| Item | Count or time | Notes |
|---|---|---|
| Plan drafts | 3 (17dc33fd, 8a82a11c, and the revision planned for after session 0, superseded by the stop) | |
| Independent reviews of the plan | 3 (design, re-review, fresh) | Each found problems the previous version missed. The fresh review found three blockers in a plan the re-review had called ready after small fixes. |
| Agent time on those reviews | about 40 minutes in all | the reviewers' reported durations: about 12, 7 and 21 minutes |

**What this says so far.** Designing a fit-first experiment carefully took three review rounds
before any data was opened. That is a cost of the route, and it counts as one.

## Sessions

| Session | Planned | Actual | Unplanned fixes | Gates failed on first run | Amendments | Ingestion hours |
|---|---|---|---|---|---|---|
| 0 | 1 session | about 0.6 h of agent wall-clock (11:07 to about 11:45 AEDT, 2026-11-14) | none to the plan (fact-finding only). Script-level: NWB `neuropal_ids` hold non-numeric strings (`merge`, 732 times); the stimulation-frame offset (0.10065 s) had to be inferred on sub-4; one operator-precedence bug in a NaN-column test, caught and rerun; one overstated missingness claim corrected before writing | none apply | none | about 0.5 h: download of the 110 segmentation files 2 min (1.70 GB); NWB reading, the re-implemented preparation over all 110 and the checks against Creamer's arrays about 25 min |
| 0b | 1 session | about 0.4 h of agent wall-clock (11:50 to about 12:12 AEDT, 2026-11-14) | none to the plan (fact-finding only). Script-level: one array-length bug in the best-ROI comparison, caught and rerun; the first Zenodo-versus-wormvae comparison skipped 10 zero-padded class names (`DB01` against `DB1`), so it was rerun with names normalised; a statement about neuron types (CEP as sensory) was corrected against `white_neuron_types.csv` before the write-up was finished. The repository's code needed three run-time shims (`np.int`, `np.NaN`, the CUDA default tensor type) to time on CPU | none apply | none | about 0.25 h: one full NWB download (1.46 GB, about 4 min); byte-range reads of the trace datasets of all 21 NWB files (55 MB, 2.5 min); byte-range extraction of three Zenodo members (124 MB, about 1 min); the mat-versus-NWB and Zenodo-versus-wormvae comparisons about 8 min |
| M-a (C.2 against measured responses) | part of a session | about 11 minutes of agent wall-clock over two runs (about 4 and 7 minutes) | two errors in my registration, both about the code: the answer vector's layout, and where stimulation enters the forward | **1:** the registered added-input unit test failed (largest hidden difference about 1.57 against 10⁻⁵, `ma_c2/parity.json`); it exposed the stimulation error | 1 (amendment 1, d9f928ac, before any measured data were read) | none |
| H0 (true wiring against 19 rewirings) | part of a session | about 2 minutes of agent wall-clock | none | none | none | none |

## After the sessions

| Item | Count or time | Notes |
|---|---|---|
| Independent check of the recommendation | 1 fresh agent, about 2 minutes of its final run | It was interrupted once by an API spend limit and resumed. Its argument led to H0. |
| Review of the merge request (PR #533) | 1 fresh agent, about 10 minutes | Two must-fix findings (a wrong provenance line in the exports' manifest, and the failed M-a unit test recorded nowhere) and several smaller ones, all applied before merge. |

## Maintainer decision rounds

Eight, counted once each, from the start of Part B to the merge:

1. the seven spike decisions (A15 to A21);
2. the approval question, answered with "one more independent review";
3. after the fresh review (A24 to A26);
4. after session 0 (A27 to A29);
5. after session 0b, the stop (A30, A31);
6. after the independent check: H0, the merge scope and the handover (A32, A33);
7. trimming the merge request;
8. applying the merge request's review.

M-a's amendment was mine, made before any data were read, and is not counted as a maintainer
round.
