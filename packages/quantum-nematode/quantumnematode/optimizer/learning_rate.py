"""Library for learning rate adjustment strategies."""

import numpy as np  # pyright: ignore[reportMissingImports]


class DynamicLearningRate:
    """
    Implements a dynamic learning rate adjustment strategy.

    The learning rate is adjusted based on the number of optimization steps taken.

    The formula used is:
        learning_rate = initial_learning_rate / (1 + decay_rate * steps)

    Attributes
    ----------
        initial_learning_rate (float): The initial learning rate.
        decay_rate (float): The decay rate for the learning rate adjustment.
        steps (int): Counter for the number of optimization steps taken.
    """

    def __init__(self, initial_learning_rate: float = 0.1, decay_rate: float = 0.01) -> None:
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.steps = 0

    def get_learning_rate(self, reward_magnitude: float = 1.0) -> float:
        """
        Compute the current learning rate.

        Compute the current learning rate based on the number of optimization steps taken
        and scale it based on the magnitude of the reward signal.

        Parameters
        ----------
        reward_magnitude : float, optional
            The magnitude of the reward signal to scale the learning rate, by default 1.0.

        Returns
        -------
            float: The current learning rate.
        """
        base_learning_rate = self.initial_learning_rate / (1 + self.decay_rate * self.steps)
        scaled_learning_rate = base_learning_rate * reward_magnitude
        self.steps += 1
        return scaled_learning_rate


class AdamLearningRate:
    """
    Implements the Adam optimization algorithm for learning rate adjustment.

    The Adam optimizer combines the advantages of two popular optimization methods:
    Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp).
    It computes adaptive learning rates for each parameter based on the first and second
    moments of the gradients.

    Attributes
    ----------
        initial_learning_rate (float): The initial learning rate for the optimizer.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): A small constant to prevent division by zero.
        m (dict): Dictionary to store the first moment estimates for each parameter.
        v (dict): Dictionary to store the second moment estimates for each parameter.
        steps (int): Counter for the number of optimization steps taken.
    """

    def __init__(
        self,
        initial_learning_rate: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.initial_learning_rate = initial_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.steps = 0

    def get_learning_rate(
        self,
        gradients: list[float],
        parameter_names: list[str],
    ) -> dict[str, float]:
        """
        Compute the effective learning rates for each parameter using Adam optimization algorithm.

        The effective learning rate is calculated using the first and
        second moment estimates of the gradients.

        Parameters
        ----------
            gradients (list[float]): List of gradients for each parameter.
            parameter_names (list[str]): List of parameter names corresponding to the gradients.

        Returns
        -------
            dict[str, float]: A dictionary mapping parameter names
                to their effective learning rates.
        """
        effective_learning_rates: dict[str, float] = {}
        for param_name, grad in zip(parameter_names, gradients, strict=False):
            if param_name not in self.m:
                self.m[param_name] = 0.0
                self.v[param_name] = 0.0

            # Reset m if the gradient sign changes
            if np.sign(grad) != np.sign(self.m[param_name]):
                self.m[param_name] = 0.0

            # Update biased first and second moment estimates
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad**2)

            # Compute bias-corrected moment estimates
            m_hat = self.m[param_name] / (1 - self.beta1 ** (self.steps + 1))
            v_hat = self.v[param_name] / (1 - self.beta2 ** (self.steps + 1))

            # Calculate effective learning rate
            effective_learning_rates[param_name] = (self.initial_learning_rate * m_hat) / (
                np.sqrt(v_hat) + self.epsilon
            )

        self.steps += 1
        return effective_learning_rates


class PerformanceBasedLearningRate:
    """
    Implements a performance-based adaptive learning rate strategy.

    The learning rate dynamically adjusts based on the agent's performance metrics,
    such as efficiency score or success rate. It increases when performance improves
    and decreases when performance stagnates or worsens.

    Attributes
    ----------
        initial_learning_rate (float): The initial learning rate.
        min_learning_rate (float): The minimum allowable learning rate.
        max_learning_rate (float): The maximum allowable learning rate.
        adjustment_factor (float): The factor by which the learning rate is adjusted.
        previous_performance (float): The performance metric from the previous step.
    """

    def __init__(
        self,
        initial_learning_rate: float = 0.1,
        min_learning_rate: float = 0.001,
        max_learning_rate: float = 0.5,
        adjustment_factor: float = 1.1,
    ) -> None:
        self.learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.adjustment_factor = adjustment_factor
        self.previous_performance = None

    def get_learning_rate(self, current_performance: float) -> float:
        """
        Adjust the learning rate based on the current performance metric.

        Parameters
        ----------
        current_performance : float
            The current performance metric (e.g., efficiency score or success rate).

        Returns
        -------
            float: The adjusted learning rate.
        """
        if self.previous_performance is not None:
            if current_performance > self.previous_performance:
                # Increase learning rate if performance improves
                self.learning_rate *= self.adjustment_factor
            else:
                # Decrease learning rate if performance stagnates or worsens
                self.learning_rate /= self.adjustment_factor

        # Constrain learning rate within allowable bounds
        self.learning_rate = max(
            self.min_learning_rate,
            min(self.max_learning_rate, self.learning_rate),
        )

        # Update previous performance
        self.previous_performance = current_performance

        return self.learning_rate
