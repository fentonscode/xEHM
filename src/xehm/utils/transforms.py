#
# Numerical transformations
#
import numpy as np

__all__ = ["mse", "mmse", "implausibility"]


# Mean square error
def mse(a, b):
    return np.square(a - b).mean()


# Maximum square error
def mmse(a, b):
    return np.square(a - b).max()


# Implausibility: This is the metric by which potential inputs are ranked
def implausibility(match: float, expectation: np.ndarray, variance: float,
                   emulator_variance: np.ndarray) -> np.ndarray:
    return np.divide(np.abs(np.subtract(match, expectation)), np.sqrt(np.add(variance, emulator_variance)))
