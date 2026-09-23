---
name: nematode-logbook
description: Create or update an experiment logbook — the claim-shaped write-up, its committed supporting data, the index row, and the roadmap, tracker and citation-site updates a result obliges. Use when the user wants to document a campaign's or evaluation's results permanently.
metadata:
  author: nematode
  version: '2.0'
---

Create or update an experiment logbook and everything a result obliges elsewhere in the repo.

**Reference logbooks.** Model new work on the most recent ones, not the early ones:
[070](../../../docs/experiments/logbooks/070-init-sharing-control.md) and
[071](../../../docs/experiments/logbooks/071-operating-point-surface.md) for a registered campaign,
[069](../../../docs/experiments/logbooks/069-phase7-synthesis.md) for a phase synthesis. Logbooks
before about 040 predate the pre-registration and retention discipline and are the wrong model.

**Input**: create or update; optionally a number, a title, or the campaign it records.

## Steps

1. **Number and scope.** Next number after `ls docs/experiments/logbooks/*.md`. If updating, name
   what new data arrives and which sections it touches.

2. **Gather what the record rests on — from files, never from terminal arithmetic.** Every figure
   the logbook states must be re-derivable from something committed. If a number exists only
   because you computed it in a shell, write it into the analysis JSON first (a driver change, a
   recorded field) and cite that. A claim resting on a manual check is the commonest way a logbook
   goes wrong.

3. **Commit the supporting data** under `docs/experiments/logbooks/supporting/<NNN>-<slug>/`:

   - `launch.md` — the pre-registration, **written before any panel seed ran**. If it is not there,
     the logbook says so rather than implying one existed.
   - the **per-seed CSV** carrying every field the analysis reads (write it with
     `lineterminator="\n"`; `csv` defaults to CRLF);
   - the **analysis JSON** the headline figures come from.

   **Retention**: those three are committed; raw campaign logs and step-level exports are archived
   off-repo, and a campaign directory is deleted only after its CSV is committed. A headline figure
   that cannot be re-derived from the committed files alone does not ship. `artifacts/` (Git LFS)
   is available for large or binary outputs such as a single evaluation's weights — optional, and
   not the place for a many-hundred-run panel.

4. **Write the logbook** from [the template](../../../docs/experiments/templates/experiment.md). The
   parts that carry the weight:

   - **A claim-shaped title and status line** — the finding, not the topic, and in the status line
     what is established, what is unresolved, and what condition now attaches.
   - **Results read against the registered branches**, with the primary reported as registered —
     an interaction where a manipulation is crossed with a structure contrast, never a main effect in
     its place. Report readings that fell below the panel's sensitivity as **unresolved at that
     sensitivity**, not as null.
   - **Corrections made in the open**: every defect found, every mid-course change to the
     protocol, and what each did to the numbers. Do not fold a correction silently into the figures.
   - **What this establishes, and what it does not** — three separate paragraphs.
   - **Registered consequences**, each with one of the five statuses: *met*, *unmet-with-reason*,
     *deferred-with-destination*, *superseded-by-result*, *unreachable-with-reason*. Deferred and
     superseded are not interchangeable.

5. **Discharge what the result obliges elsewhere, in the same change:**

   - the **index row** in `docs/experiments/README.md`, newest at the top of its block, in the
     existing format (claim-shaped title; status plus verdict and date; one long summary paragraph);
   - the **roadmap**: status rows and exit-criterion checkboxes the result moves;
   - the **phase tracker**: tick the task with its status, and name the destination of anything
     deferred — a deferred item's destination goes in **both** the tracker and the roadmap;
   - **citation sites**: where the result conditions an earlier one, add a dated note in the same
     sentence as the earlier claim at every site that cites it (roadmap, README, the earlier
     logbook, the tracker). The earlier verdict stands as read at its own setting; it is conditioned,
     not rewritten.

6. **Verify.**

   - every new relative link resolves;
   - every figure in the logbook matches the committed JSON or CSV it came from;
   - the index row, roadmap and tracker say the same thing as the logbook;
   - `uv run pre-commit run --all-files` passes, **judged by its exit code** — never by filtered
     output, which hides the failing hook and discards the status.

## Guardrails

- **Never state a number you cannot point to in a committed file.**
- **A reading below its panel's sensitivity is unresolved, not absent.** Say which.
- **Record the defect beside the figure it changed.** A logbook whose numbers quietly improved
  between drafts is less trustworthy than one that shows the correction.
- **Keep the main logbook readable**: long per-seed detail belongs in the committed CSV, and a
  `details.md` appendix is optional rather than required.
- **Match the metric to the question.** Time-to-competence and `auc_success` carry the wiring work;
  L100 is a single-configuration plateau summary from the early logbooks and is not required.
