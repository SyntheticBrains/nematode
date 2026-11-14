# Effort ledger

Every unit of work on the spike, so that Part B's estimates rest on measurement. Times are wall-clock.
"Agent" means work done by Claude or its subagents; "maintainer" means the maintainer's own reading
and decisions.

## Before session 0: designing the spike (2026-11-14)

| Item | Count or time | Notes |
|---|---|---|
| Plan drafts | 3 (17dc33fd, 8a82a11c, and the registration revision after session 0) | |
| Independent reviews of the plan | 3 (design, re-review, fresh) | Each found problems the previous version missed. The fresh review found three blockers in a plan the re-review had called ready after small fixes. |
| Maintainer decision rounds | 3 (A15 to A21, A24 to A26, and the registration) | |
| Agent time on review and revision | about 1 h of reviewer time over three reviews, plus revision | from the reviewers' reported durations: about 12, 7 and 21 minutes |

**What this says so far.** Designing a fit-first experiment carefully took three review rounds
before any data was opened. That is a cost of the route, and it counts as one.

## Sessions

| Session | Planned | Actual | Unplanned fixes | Gates failed on first run | Amendments | Ingestion hours |
|---|---|---|---|---|---|---|
| 0 | 1 session | about 0.6 h of agent wall-clock (11:07 to about 11:45 AEDT, 2026-11-14) | none to the plan (fact-finding only). Script-level: NWB `neuropal_ids` hold non-numeric strings (`merge`, 732 times); the stimulation-frame offset (0.10065 s) had to be inferred on sub-4; one operator-precedence bug in a NaN-column test, caught and rerun; one overstated missingness claim corrected before writing | none apply | none | about 0.5 h: download of the 110 segmentation files 2 min (1.70 GB); NWB reading, the re-implemented preparation over all 110 and the checks against Creamer's arrays about 25 min |
