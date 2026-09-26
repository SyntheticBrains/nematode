## Why

Shipment 8a's required items all carry a status:

- A.0 (the retention rule);
- A.1 (Logbook 070);
- A.2 (Logbook 071);
- A.4 (the methodology consolidation);
- B.1 (Logbooks 072–073);
- A.6 with its split (Logbooks 074–075).

Decision D20 makes the 8a synthesis the gate for 8b: **8b does not start until the synthesis has
given every 8a criterion one of the five statuses and written the go/no-go decision.** Nothing in 8a
needs a further run to be assigned a status.

The record is scattered. Block V's claim now carries conditions from four rungs, each added at a
different citation site on a different day. Several open threads each have a destination only in one
place. The roadmap's Phase 8 row still reads "PLANNED". The synthesis gathers these into one terminal
state for 8a.

## What Changes

- **Logbook 076, the 8a synthesis:**
  - an exit-criterion walkthrough that gives every 8a MUST, SHOULD and 8a-relevant MAY one of the five
    statuses;
  - block V restated with every condition in one sentence;
  - what 8a established, what it did not, and the cause of each negative;
  - a package inventory for A.5, not a decision;
  - a reproducibility statement;
  - the D20 gate, written as **GO**.
- **Three decisions recorded, taken before this change:**
  - B.2 is carried to 8b, with its reason;
  - A.5 is a step of its own after the synthesis;
  - **8b's wiring contrasts use the chemical-only null as primary**, with the current null reported
    beside it. This is registered as roadmap decision **D21**.
- **A protocol requirement:** a panel sized from a proxy reports its achieved sensitivity beside the
  registered one and never re-reads a verdict against it. B.1c, A.6 and the split already followed
  this practice without a rule behind it.
- **Terminal state for 8a:**
  - the roadmap's Phase 8 row reads **8a complete / 8b pending**;
  - the 8a exit criterion is ticked, and the SHOULD line and the overshoot row get their statuses;
  - the minimum-viable success level is marked met;
  - the tracker's 8a is COMPLETE and its 8b ready to start;
  - the README's status line is updated.

## Capabilities

### Modified Capabilities

- `architecture-comparison-protocol`: gains the achieved-sensitivity requirement.

## Impact

- **Documentation only:**
  - Logbook 076;
  - `docs/roadmap.md`;
  - `openspec/changes/phase8-tracking/tasks.md`;
  - the experiments index;
  - `README.md`.
- **No code, configs or campaigns.** Every figure the synthesis states comes from a committed
  logbook, CSV or JSON in 070–075.
- `phase8-tracking` stays open until the Phase 8 synthesis.
