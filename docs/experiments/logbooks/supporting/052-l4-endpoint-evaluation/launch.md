# I.1c step 0 — the endpoints, perturbation off: the registered protocol

Registered in `openspec/changes/add-l4-endpoint-evaluation`, reviewed and committed **before** the run.
Both outcomes and what each licenses were fixed there; nothing below was chosen after a result existed.

## The question

Every number the clone assay produced for the node-perturbation arm was measured **while that
arm's own perturbation was running**: 12.0 against a frozen control at 8.9 and this comparator at
38.7. That measures what the arm does while training. It cannot say what the arm *learned*,
because σ = 0.2 jitter alone costs a frozen policy ~30 points, and nothing has ever run those
endpoint weights clean.

On the MLP positive control the two differ by a lot — a policy trained at σ = 0.2 scores 86% of
the floor-to-optimum gap under that perturbation and **93%** with it off. If the connectome
endpoints behave the same way, the tension I.1 exposed is an artefact of evaluating under
exploration noise. If they do not, the rule rewrote a competent policy and the I.1 fail is a
genuine retention fail. Their cosine to the clone is 0.65–0.75, so the weights did move a long way.

## The arm

| | |
|---|---|
| config | `..._plastic_frozen_endpoint_nodeperturbation.yml` — the comparator's config with **one key changed**, `weights_path` |
| weights | the eight I.1 clone-assay endpoints, staged per seed |
| condition | `freeze_updates: true`, no eligibility mode, no perturbation scale (parent defaults `hebbian` / 0.0) — the comparator's own condition |
| seeds / budget | 1–8 × 2000 episodes |
| metric | plateau-tail full-clear %, the assay's |
| comparator | the committed frozen clone, mean **38.7** |

## The rule — the assay's, unchanged

**Holds** if the mean is within 5 points of 38.7 and ≥ 6 of 8 seeds are within 10 points of their
own clone. **Improves** if the mean exceeds 38.7 with ≥ 6 of 8 at or above their clone. Otherwise
**fails**.

- **Holds or improves** → the policy the rule left behind is competent once it runs without its
  exploration noise; the I.1 assay fail was an evaluation artefact. *Licenses:* a registered
  amendment making perturbation-off the primary endpoint of the clone assay for any perturbing
  rule, and the panel question reopening under I.2's statistic. *Does not license* the low-σ
  programme.
- **Fails** → the rule rewrote a competent policy into a worse one and removing the noise does not
  recover it; the I.1 fail stands as a genuine retention fail. *Licenses:* the low-σ programme in
  the form two critical reviews fixed it.

## The load-integrity check, read first

Updates are frozen, so each run's final weights **are** the weights it loaded, and its cosine to
the clone must reproduce the value the assay recorded for that seed:

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| recorded cosine | 0.688 | 0.725 | 0.722 | 0.678 | 0.691 | 0.748 | 0.679 | 0.649 |

A departure above 0.01 voids that seed, and **any void seed voids the verdict** — a cosine near
1.00 means the clone was evaluated rather than the endpoint, which would read as a policy that
held perfectly. Read before anything else; a void stops the scoring rather than being scored
around.

## Descriptive annotation, not a verdict input

Each seed's endpoint score against the same seed's under-perturbation score from the assay
(7.0, 24.6, 12.4, 2.8, 7.2, 3.0, 10.2, 29.0; mean 12.0) — the perturbation tax the rule was paying
at the end of training.

## Provenance

`final.pt` is the runner's end-of-run auto-save, which also fires on an interrupted run, so
completion is a precondition: **all eight source runs completed their 2000 episodes**, verified
from their logs before this was written. Staged from each run's export:

- seed 1: `exports/20260910_100830_a52766f3/weights/final.pt`
- seed 2: `exports/20260910_100831_c2f9b330/weights/final.pt`
- seed 3: `exports/20260910_100830_484a2974/weights/final.pt`
- seed 4: `exports/20260910_100830_def45a25/weights/final.pt`
- seed 5: `exports/20260910_100831_65e7a466/weights/final.pt`
- seed 6: `exports/20260910_100830_8d276728/weights/final.pt`
- seed 7: `exports/20260910_100830_916ec027/weights/final.pt`
- seed 8: `exports/20260910_100831_c80032dd/weights/final.pt`

Two episodes of seed 1 were observed while confirming an endpoint loads through the runner (both
`health_depleted`, 170 and 108 steps). Disclosed; they carry no information at n = 2 and are not
part of the record.

## Reproduce

```bash
P=configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic_frozen_endpoint_nodeperturbation
uv run python scripts/run_campaign.py --config ${P}.yml --seeds 1-8 --runs 2000 \
  --output-dir campaigns/l4-endpoint-evaluation -- --theme headless --track-experiment
```
