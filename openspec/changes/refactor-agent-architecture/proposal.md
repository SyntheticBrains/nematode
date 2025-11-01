# Refactor Agent Architecture

## Summary
Refactor the monolithic `QuantumNematodeAgent` class into a modular, testable architecture by decomposing the 842-line class into focused, single-responsibility components. The current implementation contains two massive methods (`run_episode` with 268 lines and `run_manyworlds_mode` with 192 lines) that violate clean code principles, duplicate logic extensively, and make comprehensive testing extremely difficult.

## Motivation
The current `QuantumNematodeAgent` implementation has become a maintenance burden:

1. **Untestable Code**: Coverage is effectively 0% for the agent module because the large methods contain complex control flow that's difficult to test in isolation
2. **Code Duplication**: Brain parameter construction, reward calculation, food consumption logic, and rendering code is duplicated between `run_episode` and `run_manyworlds_mode`
3. **Mixed Responsibilities**: The class handles episode orchestration, rendering, state management, metrics tracking, satiety management, and reward calculation all in one place
4. **Poor Extensibility**: Adding new episode modes or step behaviors requires modifying the monolithic methods
5. **Environment-Specific Logic**: Scattered `isinstance(self.env, DynamicForagingEnvironment)` checks violate the open-closed principle

## Goals
- Achieve >70% test coverage for agent-related code through better testability
- Eliminate code duplication between episode execution modes
- Enable independent testing of episode logic, step execution, metrics, and rendering
- Support future episode modes without modifying existing code (strategy pattern)
- Separate environment-specific behavior from core agent logic

## Non-Goals
- Changing the external API or behavior of `QuantumNematodeAgent` (maintain backward compatibility)
- Modifying brain architectures or environment implementations
- Adding new features beyond the refactoring itself
- Performance optimization (maintain current performance characteristics)

## Success Criteria
- All existing tests pass without modification
- Test coverage for agent module increases from ~0% to >70%
- `run_episode` and `run_manyworlds_mode` methods reduced to <50 lines each (orchestration only)
- No duplicated logic between episode execution modes
- Pyright and ruff compliance maintained

## Stakeholders
- Research team using the agent for quantum ML experiments
- Developers adding new episode modes or environment types
- Test engineers improving code coverage

## Risks and Mitigations
**Risk**: Breaking existing simulations or changing behavior
**Mitigation**: Maintain exact backward compatibility, comprehensive integration tests

**Risk**: Over-engineering with too many small classes
**Mitigation**: Follow "rule of three" - only extract when duplication exists or clear SRP violation

**Risk**: Performance regression from additional abstraction layers
**Mitigation**: Keep hot-path code inline, measure before/after performance

## Timeline
- Estimated effort: 3-5 days
- Can be implemented incrementally without breaking existing functionality
- Each component can be extracted and tested in isolation before integration
