# Design: Refactor Agent Architecture

## Overview
This design document outlines the decomposition of the monolithic `QuantumNematodeAgent` class into a set of focused, testable components following SOLID principles.

## Current Architecture Problems

### Code Metrics
- **Total lines**: 842 (one of the largest files in the codebase)
- **`run_episode` method**: 268 lines (violates cyclomatic complexity limits)
- **`run_manyworlds_mode` method**: 192 lines (violates cyclomatic complexity limits)
- **Test coverage**: ~0% (effectively untestable in current form)
- **Noqa directives**: Multiple `C901, PLR0912, PLR0915` (complexity warnings suppressed)

### Duplication Analysis
Duplicated code patterns between `run_episode` and `run_manyworlds_mode`:

1. **Brain parameter construction** (3 instances):
   ```python
   params = BrainParams(
       gradient_strength=gradient_strength,
       gradient_direction=gradient_direction,
       agent_direction=self.env.current_direction,
       action=ActionData(action=top_action, reward=reward),
   )
   ```

2. **Food consumption logic** (2 instances in each method):
   - Distance tracking for dynamic environments
   - Satiety restoration
   - Reward calculation
   - Metrics updates

3. **Rendering logic** (2 instances):
   - Clear screen conditionals
   - Grid rendering
   - Frame display

4. **Environment-specific conditionals** (scattered throughout):
   - `isinstance(self.env, DynamicForagingEnvironment)` checks
   - Dynamic vs static reward calculations

## Proposed Architecture

### Component Hierarchy

```text
QuantumNematodeAgent (Facade/Orchestrator)
├── StepProcessor (Single step execution)
│   ├── prepare_brain_params()
│   ├── execute_action()
│   └── process_step_result()
├── FoodConsumptionHandler (Food interaction logic)
│   ├── check_food_reached()
│   ├── consume_food()
│   └── update_food_metrics()
├── SatietyManager (Hunger/satiety system)
│   ├── decay_satiety()
│   ├── restore_satiety()
│   └── check_starvation()
├── MetricsTracker (Performance tracking)
│   ├── track_step()
│   ├── track_food_collection()
│   └── calculate_final_metrics()
├── EpisodeRenderer (Visualization)
│   ├── should_render_frame()
│   ├── render_frame()
│   └── clear_screen()
└── EpisodeRunner (Strategy Pattern)
    ├── StandardEpisodeRunner
    └── ManyworldsEpisodeRunner
```

### Design Patterns Applied

#### 1. Strategy Pattern (Episode Runners)
**Problem**: Different episode execution modes with duplicated orchestration logic

**Solution**: Abstract episode execution into strategy classes
```python
class EpisodeRunner(Protocol):
    def run(
        self,
        agent: QuantumNematodeAgent,
        reward_config: RewardConfig,
        max_steps: int,
        **kwargs,
    ) -> EpisodeResult:
        ...

class StandardEpisodeRunner:
    """Runs a single trajectory episode."""

class ManyworldsEpisodeRunner:
    """Runs many-worlds branching episode."""
```

**Benefits**:
- Easy to add new episode modes (e.g., multi-agent, hierarchical)
- Testable in isolation
- No code duplication between modes

#### 2. Single Responsibility Principle (Component Extraction)

**StepProcessor**
- **Responsibility**: Execute a single simulation step
- **Public API**:
  ```python
  def process_step(
      self,
      gradient_strength: float,
      gradient_direction: float,
      previous_action: Action | None,
      previous_reward: float,
  ) -> StepResult
  ```
- **Benefits**: Testable without full episode, reusable across episode modes

**FoodConsumptionHandler**
- **Responsibility**: Handle all food-related logic
- **Public API**:
  ```python
  def check_and_consume_food(
      self,
      agent_pos: tuple[int, int],
  ) -> FoodConsumptionResult
  ```
- **Benefits**: Encapsulates environment-specific food logic

**SatietyManager**
- **Responsibility**: Manage hunger/satiety system
- **Public API**:
  ```python
  def decay(self, amount: float) -> float
  def restore(self, amount: float) -> float
  def is_starved(self) -> bool
  ```
- **Benefits**: Testable state machine, clear hunger mechanics

**MetricsTracker**
- **Responsibility**: Track and calculate episode metrics
- **Public API**:
  ```python
  def track_step(self, step_data: StepData) -> None
  def track_food_collection(self, efficiency: float) -> None
  def calculate_metrics(self, total_runs: int) -> PerformanceMetrics
  ```
- **Benefits**: Clean separation of business logic from metrics

**EpisodeRenderer**
- **Responsibility**: Handle all visualization logic
- **Public API**:
  ```python
  def render_if_needed(
      self,
      env: Environment,
      step: int,
      max_steps: int,
      show_last_frame_only: bool,
  ) -> None
  ```
- **Benefits**: Rendering logic doesn't pollute episode execution

#### 3. Dependency Injection
**Problem**: Hard to test components that directly access environment

**Solution**: Inject dependencies through constructors
```python
class StepProcessor:
    def __init__(
        self,
        brain: Brain,
        env: Environment,
        food_handler: FoodConsumptionHandler,
        satiety_manager: SatietyManager,
    ):
        self.brain = brain
        self.env = env
        self.food_handler = food_handler
        self.satiety_manager = satiety_manager
```

**Benefits**: Easy to mock dependencies in tests, clear component boundaries

### Backward Compatibility Strategy

The `QuantumNematodeAgent` class will remain as a facade that delegates to the new components:

```python
class QuantumNematodeAgent:
    """
    Quantum nematode agent (refactored facade).

    This class maintains backward compatibility while delegating
    to smaller, testable components internally.
    """

    def __init__(self, brain: Brain, env: Environment, ...):
        # Initialize new components
        self.satiety_manager = SatietyManager(...)
        self.metrics_tracker = MetricsTracker()
        self.food_handler = FoodConsumptionHandler(env, satiety_manager)
        self.step_processor = StepProcessor(brain, env, food_handler, satiety_manager)
        self.renderer = EpisodeRenderer()

        # Keep existing public attributes for compatibility
        self.brain = brain
        self.env = env
        ...

    def run_episode(self, ...) -> list[tuple]:
        """Delegates to StandardEpisodeRunner."""
        runner = StandardEpisodeRunner(
            step_processor=self.step_processor,
            metrics_tracker=self.metrics_tracker,
            renderer=self.renderer,
        )
        result = runner.run(self, ...)
        return result.path  # Maintain return type compatibility

    def run_manyworlds_mode(self, ...) -> list[tuple]:
        """Delegates to ManyworldsEpisodeRunner."""
        runner = ManyworldsEpisodeRunner(
            step_processor=self.step_processor,
            metrics_tracker=self.metrics_tracker,
            renderer=self.renderer,
        )
        result = runner.run(self, ...)
        return result.path  # Maintain return type compatibility
```

### Data Transfer Objects

New DTOs to support clean interfaces:

```python
@dataclass
class StepResult:
    """Result of processing a single step."""
    action: Action
    reward: float
    done: bool
    info: dict[str, Any]

@dataclass
class FoodConsumptionResult:
    """Result of checking/consuming food."""
    food_consumed: bool
    satiety_restored: float
    reward: float
    distance_efficiency: float | None  # For dynamic environments

@dataclass
class EpisodeResult:
    """Complete result of an episode."""
    path: list[tuple[int, int]]
    success: bool
    total_reward: float
    steps_taken: int
    metrics: dict[str, Any]
```

## Testing Strategy

### Unit Tests (New)
Each component will have comprehensive unit tests:

1. **StepProcessor Tests** (~15 tests)
   - Brain parameter construction
   - Action execution
   - State updates
   - Edge cases (stuck agent, invalid actions)

2. **FoodConsumptionHandler Tests** (~10 tests)
   - Food detection
   - Satiety restoration
   - Distance efficiency calculation (dynamic env)
   - Static vs dynamic environment behavior

3. **SatietyManager Tests** (~8 tests)
   - Decay mechanics
   - Restoration mechanics
   - Starvation detection
   - Boundary conditions (min/max satiety)

4. **MetricsTracker Tests** (~12 tests)
   - Step tracking
   - Food collection tracking
   - Metrics calculation
   - Efficiency calculations

5. **EpisodeRenderer Tests** (~6 tests)
   - Render timing logic
   - Last frame only mode
   - Screen clearing logic

6. **EpisodeRunner Tests** (~20 tests)
   - Standard episode execution
   - Many-worlds episode execution
   - Episode termination conditions
   - Integration with components

**Total**: ~71 new unit tests (expected coverage >70%)

### Integration Tests (Existing)
- Keep all existing integration tests unchanged
- Add regression tests to ensure refactored behavior matches original

### Performance Tests
- Benchmark episode execution before/after
- Ensure no >5% performance regression

## Migration Path

### Phase 1: Extract Pure Functions
- Extract brain parameter construction
- Extract food consumption logic
- Extract satiety calculations
- **No breaking changes**, just internal refactoring

### Phase 2: Create Component Classes
- Create StepProcessor, FoodConsumptionHandler, etc.
- Keep existing methods as wrappers
- Add unit tests for new components

### Phase 3: Create Episode Runners
- Implement StandardEpisodeRunner
- Implement ManyworldsEpisodeRunner
- Update run_episode/run_manyworlds_mode to delegate

### Phase 4: Cleanup and Documentation
- Remove old code once delegation is confirmed working
- Update docstrings
- Add architecture documentation

## Alternative Approaches Considered

### Alternative 1: Keep Monolithic Class
**Pros**: No refactoring effort, no risk of behavior change
**Cons**: Impossible to test, unmaintainable, blocks future development
**Decision**: Rejected - technical debt is too high

### Alternative 2: Full Rewrite
**Pros**: Clean slate, modern design from scratch
**Cons**: High risk, long timeline, likely to introduce bugs
**Decision**: Rejected - incremental refactoring is safer

### Alternative 3: Only Extract Helper Functions
**Pros**: Minimal changes, low risk
**Cons**: Doesn't solve testability or extensibility issues
**Decision**: Rejected - insufficient improvement

## Open Questions

1. **Should we support custom episode runners?**
   - Leaning yes - enables research experiments
   - Would require stable StepProcessor API

2. **Should metrics be streamed or batch calculated?**
   - Current: Batch (calculate at end)
   - Alternative: Streaming (update incrementally)
   - Leaning batch for simplicity

3. **How to handle environment-specific logic long-term?**
   - Current proposal: FoodConsumptionHandler abstracts it
   - Alternative: Environment protocol methods
   - Needs discussion with team

## References
- Current implementation: `packages/quantum-nematode/quantumnematode/agent.py`
- Related specs: `brain-architecture`, `configuration-system`
- Test coverage report: Currently ~0% for agent module
