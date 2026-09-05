# Spec: cli-interface

## ADDED Requirements

### Requirement: Accelerator selection and availability validation

The device options offered by the CLI SHALL correspond to backends the running build can actually provide, and a request for an unavailable accelerator SHALL fail at startup with an actionable message rather than an unhandled error raised later during brain construction.

`DeviceType` SHALL expose a `mps` member mapping to PyTorch's `"mps"` device string, so Apple GPUs are selectable and therefore measurable. `gpu` SHALL continue to map to `"cuda"` and SHALL NOT be silently redirected to another accelerator: a redirect would deliver a materially different performance profile under a name the user chose deliberately.

When a selected accelerator is unavailable in the running build, the CLI SHALL raise a clear error before brain construction that names the requested device, states that it is unavailable, and names the platform-appropriate alternative.

#### Scenario: Requesting CUDA on a build without CUDA

- **GIVEN** a PyTorch build without CUDA support
- **WHEN** the CLI is invoked with the `gpu` device
- **THEN** it SHALL exit with a clear error naming the unavailable device and the platform-appropriate alternative
- **AND** SHALL NOT surface a raw torch assertion from inside brain construction

#### Scenario: Requesting MPS where it is available

- **GIVEN** a build where the MPS backend is available
- **WHEN** the CLI is invoked with the `mps` device
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
