## Why

The current QVarCircuitBrain suffers from barren-plateau-like behavior where parameter-shift gradients perform poorly (22.5% success vs 88% with CMA-ES evolutionary optimization). QRC (Quantum Reservoir Computing) inherently avoids barren plateaus by using a fixed quantum reservoir with only a classical readout layer trained—providing a simpler, more reliable quantum architecture for the Phase 2 comparative analysis.

## What Changes

- Add new `QRCBrain` architecture implementing Quantum Reservoir Computing
- Fixed random quantum reservoir circuit (Hadamard + random rotations + entangling layers)
- Trainable classical readout network (MLP or linear) using standard gradient descent
- Sensory input encoding via rotation gates on reservoir qubits
- Reservoir state extraction via measurement probability distribution
- Integration with existing brain factory, configuration system, and benchmarking infrastructure
- Example YAML configurations for QRC on foraging and predator tasks

## Capabilities

### New Capabilities

- `qrc-brain`: Quantum Reservoir Computing brain architecture with fixed quantum reservoir and trainable classical readout layer

### Modified Capabilities

- `brain-architecture`: Add QRC to the brain factory and CLI brain selection (extending existing spec with new brain type)

## Impact

- **Code**: New `qrc.py` in `brain/arch/`, updates to `brain_factory.py`, new Pydantic config class
- **APIs**: Brain protocol compliance required (ClassicalBrain interface, ActionData/BrainData compatibility)
- **Dependencies**: Uses existing Qiskit infrastructure (QuantumCircuit, AerSimulator)
- **Configs**: New `qrc_foraging_small.yml` and `qrc_predators_small.yml` example configs
- **Tests**: Unit tests for reservoir circuit generation, readout training, integration tests for full episodes
- **Benchmarks**: New benchmark categories for QRC performance comparison

## Post-Implementation Notes

Implementation complete with additional enhancements (data re-uploading, entropy regularization, orthogonal init). Performance evaluation showed 0% success on foraging tasks—the fixed reservoir doesn't create discriminative representations for chemotaxis. See Logbook 008 for full analysis. Architecture may suit time-series or richer-input tasks.
