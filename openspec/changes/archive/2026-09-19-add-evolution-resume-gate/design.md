# Design: why a gate, and why the tests have to fight the test harness

## The gate is on the invocation, not the file

Nothing about a checkpoint file distinguishes one this machine wrote from one that arrived by email.
A magic byte, a checksum, a signature — each would have to be produced by the same code path an
attacker controls when they hand you the file. So the only place a real control can sit is the
**invocation**: the caller knows the provenance, and the flag makes them assert it.

That is why this is not a warning. A warning in `--help` is read once, at most, by someone who is not
at that moment thinking about provenance. A required flag is read at the moment it matters.

## Where the gate sits, and why that is load-bearing

Immediately after argument parsing, before `_resolve_output_dir` and before the loop is constructed.
Two consequences the tests pin:

- **It fires before any file access.** A missing checkpoint and a hostile checkpoint produce the same
  refusal, because the gate never looks at the path. If the gate sat after the existence check, a
  hostile file would first be *stat*ed and the error ordering would tell an attacker their file was
  found.
- **It fires before `pickle.load`.** Verified with a payload that writes a file when unpickled: without
  the flag the file is never written and the load is never reached; with the flag the traceback shows
  `pickle.load` being entered, which is what makes the flag the thing that permits the load rather
  than decoration around it.

## The refusal must be printed, and the reason is a trap worth recording

`quantumnematode.logging_config` decides at **import** whether it is running under test. In test mode
it calls `basicConfig`, which installs a stream handler. In production it installs a `NullHandler` and
defers real handlers to `configure_file_logging()`, which the driver calls late — well after this gate.

So a refusal written with `logger.error` is **silent in production**. The first draft of this change did
exactly that, and its tests passed, because `PYTEST_CURRENT_TEST` is in the environment of any
subprocess a test spawns: the child detected pytest, installed a stream handler, and the message
appeared. The gate would have shipped as a bare `exit 1`.

Two things follow, and both are in the change:

1. The gate **prints to stderr**.
2. The tests **strip `PYTEST_CURRENT_TEST` and `TESTING`** from the child's environment, so they
   exercise the production logging path. One test asserts the refusal is visible with no handler
   configured — which is the test that would have caught the first draft.

This is a general hazard in this repository: any early-exit message in an entry point, before
`configure_file_logging()`, is invisible to users and visible to tests. Other such messages exist
(`"Checkpoint not found"` among them) and are **not** fixed here; they are pre-existing, wider than
this change, and worth their own issue.

## Why the spec needs modifying rather than only extending

`evolution-framework` § Evolution Loop Checkpoint and Resume says:

> **WHEN** the run is invoked with `--resume <path>` **THEN** the loop SHALL resume from generation 6

That is now false on its own: the flag is also required. Adding a new requirement beside it would leave
two requirements disagreeing, and the one a reader hits first would be the wrong one. So the scenario
is modified to carry the flag, and a new requirement states the gate's own obligations — default off,
refuse before unpickling, and be visible without a configured log handler.

## What a reviewer should check

That the gate cannot be satisfied by anything other than the flag; that no call site in the repository
still invokes `--resume` without it; and that the new requirement's visibility clause is actually
tested against the production logging path rather than the test-mode one.
