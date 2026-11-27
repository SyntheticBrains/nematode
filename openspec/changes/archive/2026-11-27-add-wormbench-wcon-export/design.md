## Decisions

- **A new module, not an edit to `track_import`.** The writers are the importer's inverse, but
  `track_import` is grading code, and an edit to it would call for the byte-identity check across
  trees. A new module leaves every grading path untouched.
- **Bytes as the script wrote them.** The worked example's archive is built through
  `track_import_check.write_track`, and `wormbench_example.py check` compares its bytes. The package's
  CSV writer therefore keeps the script's format: Python's default CSV dialect, `repr` for every
  number (a missing sample is `nan`), and a compact sidecar. The WCON file is compact JSON with a
  trailing newline; `null` is a missing sample.
- **The conversion checks itself.** `wormbench wcon` writes a staged file beside the output, reads it
  back with the importer and compares every animal's times and positions exactly, and with
  `--simulated` its labels and seed, and the metadata. Only then does the staged file replace the
  output; on a difference the output is left as it was. Exact comparison holds because JSON and
  `repr` both round-trip a float. An output that would overwrite the source, or that the grader
  would read as a CSV by its suffix, is refused before anything is read.
- **CSV in, WCON out.** WCON input is not converted: the importer reduces a skeleton to its middle
  point, so rewriting a WCON file would discard data the user may want, for no gain in grading.
- **The on-lawn flag is refused, not translated.** WCON's core has no field for it, and a custom
  `@` field would be dropped by every reader but this one. A lawn family's track that carries the
  flag is refused by name, and the user gives the boundary, or `"uniform"`, in the sidecar instead.
  Local search ignores lawn membership, so its flag is dropped.

## Checks made during development

- The worked example's 64 animals, converted by `wormbench wcon`, grade identically to the CSVs on
  v4 (a test).
- The 64 converted files, and a simulated file with `@primitive` and `@seed`, validate against WCON
  1.0's JSON schema (`wcon_schema.json` from openworm/tracker-commons, draft-07), and the same
  validator rejects a file without `units` and one with a malformed time array. The schema is not
  vendored, since a test would need `jsonschema`.
- `track_import_check.py` on the package's writers writes a record byte-identical to the committed
  `supporting/115-benchmark-v2/import_roundtrip_v2.json`, and `wormbench_example.py check` passes.
- Eleven planted defects each fail at least one test: a missing sample written as 0, the on-lawn
  refusal removed, the seed dropped, the lawn centre dropped, the sample check disabled, the
  metadata check disabled, repeated animals allowed, positions rounded to six places, a simulated
  CSV's labels and seeds lost, every on-lawn flag written as 1, and the staged file never moved into
  place.
- An independent review found, and the change now refuses: a track with no samples (which the CSV
  route dropped and the WCON route crashed on), ids 1 and "1" (one animal to the importer), simulated
  tracks the importer's simulated route refuses, metadata shapes that read back otherwise, and an
  output path equal to the source. It also found three untested branches, now tested.
