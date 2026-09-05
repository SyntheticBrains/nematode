# Spec: cli-interface

## ADDED Requirements

### Requirement: Accelerator selection and availability validation

The device options offered by the CLI SHALL correspond to backends the running build can actually provide, and a request for an unavailable accelerator SHALL fail at startup with an actionable message rather than an unhandled error raised later during brain construction.

`DeviceType` SHALL expose a `mps` member mapping to PyTorch's `"mps"` device string, so Apple GPUs are selectable and therefore measurable. `gpu` SHALL continue to map to `"cuda"` and SHALL NOT be silently redirected to another accelerator: a redirect would deliver a materially different performance profile under a name the user chose deliberately.

When a selected accelerator is unavailable in the running build, the CLI SHALL raise a clear error before brain construction that names the requested device, states that it is unavailable, and names the platform-appropriate alternative.

The availability check SHALL apply to **every** brain, including those in the `quantum` family. A `quantum` tag does not imply a brain is Qiskit-only: most quantum brains also construct PyTorch actors and critics and place those tensors on the selected device, so exempting them would reinstate the very unhandled construction-time error this requirement exists to replace. A host offering a GPU-enabled Qiskit simulator with a deliberately CPU-only PyTorch build is refused by this check; that is the accepted trade, and it fails with a message naming the problem rather than a traceback from inside a brain.

#### Scenario: Requesting CUDA on a build without CUDA

- **GIVEN** a PyTorch build without CUDA support
- **WHEN** the CLI is invoked with the `gpu` device
- **THEN** it SHALL exit with a clear error naming the unavailable device and the platform-appropriate alternative
- **AND** SHALL NOT surface a raw torch assertion from inside brain construction

#### Scenario: Quantum brains that build PyTorch modules are checked too

- **GIVEN** a PyTorch build without CUDA support
- **WHEN** the CLI is invoked with the `gpu` device and a brain in the `quantum` family that constructs PyTorch modules
- **THEN** it SHALL exit with a clear error rather than failing during brain construction

#### Scenario: A device that places no tensors needs no availability check

- **WHEN** the CLI is invoked with `qpu`, which routes to a quantum backend rather than placing tensors on an accelerator
- **THEN** no PyTorch availability check SHALL be applied

#### Scenario: Requesting MPS where it is available

- **GIVEN** a build where the MPS backend is available
- **WHEN** the CLI is invoked with the `mps` device and a brain that runs on the PyTorch backend
- **THEN** brain tensors SHALL be placed on the MPS device
- **AND** the run SHALL proceed without error

#### Scenario: Requesting MPS where it is unavailable

- **GIVEN** a build where the MPS backend is unavailable
- **WHEN** the CLI is invoked with the `mps` device
- **THEN** it SHALL exit with a clear error naming the unavailable device
- **AND** SHALL NOT surface a raw torch error from inside brain construction

#### Scenario: CPU remains the default

- **WHEN** no device is specified
- **THEN** the CLI SHALL select CPU
- **AND** the selection SHALL require no accelerator availability check

### Requirement: Device selection is validated against the brain family

`DeviceType` is shared by two unrelated backends: the PyTorch device for classical/spiking brains, and the Qiskit Aer / IBM Runtime backend selector for quantum brains, which reach it as an upper-cased string. A device meaningful to one backend is not necessarily meaningful to the other, and the quantum backend accepts unknown device strings **without raising**, so an unchecked selection is recorded in experiment metadata as though it were a legitimate backend.

The CLI SHALL therefore reject, at startup and before brain construction, any device that the selected brain's family cannot use. Specifically, a PyTorch-only accelerator SHALL NOT be accepted for a brain registered in the `quantum` family, whose device value is consumed as a simulator selector rather than a tensor placement. The error SHALL name the requested device, the brain, and the devices that brain does accept.

The brain family SHALL be read from the existing plugin-registry `families` metadata rather than from a hand-maintained list, so a newly registered architecture inherits the correct validation without a per-architecture branch. A brain tagged with both `quantum` and another family SHALL be treated as quantum for this check, because its device value still reaches the simulator.

This requirement adds validation only for accelerators introduced by this change; it SHALL NOT alter the pre-existing tolerance whereby a non-quantum brain selecting `qpu` falls back to CPU tensor placement.

#### Scenario: PyTorch-only accelerator rejected for a quantum brain

- **GIVEN** a config selecting a brain registered in the `quantum` family
- **WHEN** the CLI is invoked with the `mps` device
- **THEN** it SHALL exit with a clear error naming the device, the brain, and the accepted devices
- **AND** SHALL NOT construct a simulator backend from the rejected device string

#### Scenario: Hybrid quantum brains are treated as quantum

- **GIVEN** a brain whose registry entry tags it both `quantum` and `classical`
- **WHEN** the CLI is invoked with a PyTorch-only accelerator
- **THEN** the selection SHALL be rejected on the same grounds as a purely quantum brain

#### Scenario: Quantum brains keep their own devices

- **WHEN** a quantum brain is run with `cpu`, `gpu`, or `qpu`
- **THEN** the selection SHALL be accepted
- **AND** the simulator backend string SHALL be unchanged from before this change

#### Scenario: Validation derives from registry metadata

- **WHEN** a new architecture is registered with a `quantum` family tag
- **THEN** it SHALL be covered by this validation without any change to the validation code

#### Scenario: Every agent of a multi-agent run is validated

- **GIVEN** a multi-agent configuration whose agents name different architectures
- **WHEN** a device is selected that one of those agents cannot use
- **THEN** the run SHALL be rejected before any brain is constructed
- **AND** the error SHALL identify which agent is at fault
- **AND** a multi-agent configuration whose agents can all use the device SHALL be accepted

#### Scenario: Multi-agent runs cannot bypass validation

- **WHEN** a multi-agent run is dispatched
- **THEN** device validation SHALL have been applied to every configured agent's brain, not only to a single top-level brain
