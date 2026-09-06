# Tasks: freeze the matched-rule yardstick's readout

## 1. The option

- [x] 1.1 `plastic_layers: Literal["all", "hidden"] = "all"` on the MLP brain config, passed to the
  topology at construction.
- [x] 1.2 `MLPTopology` builds its plastic list from the setting: every `Linear` under `all`, every
  `Linear` but the last under `hidden`; masks, traces and fan-in axes follow the list; the forward
  runs the whole actor and credits eligibility to plastic layers only; the `layers` property and its
  docstring say it holds the plastic layers, which under `hidden` exclude the output layer.

## 2. Tests

- [x] 2.1 `all` is byte-identical: the seam lists, the trace buffers and the forward are today's (the
  existing MLP seam, equivalence and matched-rule tests keep passing).
- [x] 2.2 `hidden`: the seam has one entry fewer; no trace buffer for the output layer; the forward
  output is bitwise-equal to `actor(features)`; after rule steps the output weight and bias are
  bit-identical and the hidden weights are not; homeostatic targets exist for hidden layers only.
- [x] 2.3 The PPO path is unaffected by the option (`learnable_parameters` unchanged; the frozen
  reference equivalence holds with `hidden` set).

## 3. Docs and close-out

- [x] 3.1 `docs/architectures.md`: the plasticity row's "every Linear weight is plastic including the
  output layer — a deliberate asymmetry in the MLP's favour" is reworded to the selectable depth and
  its reason; `configs/README.md`'s "every Linear weight plastic" likewise; `CHANGELOG.md`.
- [x] 3.2 Pre-commit gate on all files exit 0; full suite green. (4610 passed as CI runs it)
- [x] 3.3 No implementation code or docstring references a planning document.
- [x] 3.4 Re-review for drift, archive, review the branch, open the PR. (every new scenario maps to a named test; the kept scenarios to the existing MLP suites)
