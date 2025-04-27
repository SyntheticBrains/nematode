import numpy as np


class DynamicLearningRate:
    def __init__(self, initial_learning_rate: float = 0.1, decay_rate: float = 0.01):
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.steps = 0

    def get_learning_rate(self):
        learning_rate = self.initial_learning_rate / (1 + self.decay_rate * self.steps)
        self.steps += 1
        return learning_rate


class AdamLearningRate:
    def __init__(self, initial_learning_rate: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.initial_learning_rate = initial_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.steps = 0

    def get_learning_rate(self, gradients, parameter_names):
        effective_learning_rates = {}
        for param_name, grad in zip(parameter_names, gradients):
            if param_name not in self.m:
                self.m[param_name] = 0
                self.v[param_name] = 0

            # Update biased first and second moment estimates
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected moment estimates
            m_hat = self.m[param_name] / (1 - self.beta1 ** (self.steps + 1))
            v_hat = self.v[param_name] / (1 - self.beta2 ** (self.steps + 1))

            # Calculate effective learning rate
            effective_learning_rates[param_name] = (self.initial_learning_rate * m_hat) / (np.sqrt(v_hat) + self.epsilon)

        self.steps += 1
        return effective_learning_rates