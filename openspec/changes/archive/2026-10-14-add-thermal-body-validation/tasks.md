# Tasks: thermotaxis through the body, and M.9's speed–stability split

- [x] 1. **Harness**: `run_capture` takes the capture modality. A test that a thermal evaluation
  records `−|T − Tc|`.
- [x] 2. **Analysis**: `thermal_body.py thermotaxis` evaluates the four arms on seeds 2301–2332 (30
  held-out episodes each). It grades each arm with the thermal reference set and C.3's thresholds,
  compares each trained arm with its floor and describes the wild type minus the null. Tests.
- [x] 3. **Run and read**: commit the per-arm grades, the floor comparison and the per-run JSON.
  Write Logbook 093 and the speed–stability addendum to Logbook 092, with its JSON.
- [x] 4. **Records**: the tracker's C.3 thermotaxis note and M.9 note, the experiment index, and the
  CHANGELOG.
- [x] 5. **Close-out**: the full suite, hooks by exit code, validate, archive, then the PR.
