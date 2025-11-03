import builtins

from quantumnematode.report import summary as summary_mod
from quantumnematode.report.dtypes import SimulationResult


class DummyLogger:
    """A dummy logger to capture log messages for testing."""

    def __init__(self):
        self.disabled = False
        self.infos = []

    def debug(self, msg):
        """Capture debug messages."""
        self.infos.append(msg)

    def info(self, msg):
        """Capture info messages."""
        self.infos.append(msg)


def make_sim_result(run, steps, efficiency_score, path=None):
    """
    Create a dummy SimulationResult for testing.

    Parameters
    ----------
    run : int
        The run number.
    steps : int
        The number of steps taken in the simulation.
    efficiency_score : float
        The efficiency score of the simulation.
    path : list[tuple[int, int]], optional
        The path taken in the simulation, defaults to None.

    Returns
    -------
    SimulationResult
        A SimulationResult object with the provided parameters.
    """
    if path is None:
        path = [(0, 0), (1, 1)]
    return SimulationResult(
        run=run,
        steps=steps,
        efficiency_score=efficiency_score,
        path=path,
        total_reward=0.0,
        last_total_reward=0.0,
    )


def test_summary_print_and_logger(monkeypatch):
    """Test that the summary function prints and logs the expected output."""
    # Prepare dummy results
    results = [
        make_sim_result(1, 100, 0.8),
        make_sim_result(2, 80, 0.9),
        make_sim_result(3, 60, 1.0),
    ]
    num_runs = 3
    max_steps = 120
    dummy_logger = DummyLogger()
    monkeypatch.setattr(summary_mod, "logger", dummy_logger)
    printed = []
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: printed.append(args))
    summary_mod.summary(num_runs, max_steps, results)
    # Check print output
    assert any("Average steps per run:" in str(x) for x in printed)
    assert any("Success rate:" in str(x) for x in printed)
    # Check logger output
    assert any("Average steps per run:" in msg for msg in dummy_logger.infos)
    assert any("Simulation completed." in msg for msg in dummy_logger.infos)


def test_summary_logger_disabled(monkeypatch):
    """Test that the summary function does not log when the logger is disabled."""
    results = [make_sim_result(1, 10, 0.5), make_sim_result(2, 5, 0.7)]
    num_runs = 2
    max_steps = 20
    dummy_logger = DummyLogger()
    dummy_logger.disabled = True
    monkeypatch.setattr(summary_mod, "logger", dummy_logger)
    printed = []
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: printed.append(args))
    summary_mod.summary(num_runs, max_steps, results)
    # Logger should not be called
    assert dummy_logger.infos == []
    # Print output should still be present
    assert any("Average steps per run:" in str(x) for x in printed)
