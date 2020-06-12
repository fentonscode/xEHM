import numpy as np


# Draws uniform samples from an n-dimensional hypercube
# limits: min/max limits for each variable - [2 x dims] matrix, row 1 = min, row 2 - max
# NOTE: np.random.uniform adds extra container fluff - don't squeeze or we break 1D uses
def uniform_box(limits: np.ndarray, num_rows=1) -> np.ndarray:
    dims: int = limits.shape[1]
    rand = np.random.uniform
    return np.asarray([rand(low=limits[0, i], high=limits[1, i], size=(num_rows, 1))
                      for i in range(dims)]).transpose()[0]


# Exploit numpy's operators to check if a collection of points are within limits
# x should be squeezed to eliminate spurious dimensions
# x: [n x dims] test list of points
# limits: [2 x dims] matrix as columns of min/max for each dimension
def point_in_box(x: np.ndarray, limits: np.ndarray) -> bool:
    return not (np.any(x < limits[0, :]) or np.any(x > limits[1, :]))


# Generates a hypercube of samples based on the axes and size supplied
# Axes: List of tuples of (start, stop) pairs - e.g. [(0.0, 1.0), (-1.0, 1.0)]
# Size: number of samples (same across axes)
# TODO: This is failing for 1D inputs - do we need a reshape operation in here?
def hypercube(axes, size):
    ndims = len(axes)
    rev_axes = axes[::-1]
    L = [np.linspace(rev_axes[i][0], rev_axes[i][1], size) for i in range(ndims)]
    return np.flip(np.hstack((np.meshgrid(*L))).swapaxes(0, 1).reshape(ndims, -1).T, axis=1)


# Scale matrix columns to a min/max range
# matrix: [r x c] ndarray
# scales: min/max limits for each variable - [2 x c] ndarray, row 1 = min, row 2 - max
def scale_matrix_columns(matrix: np.ndarray, scales: np.ndarray) -> np.ndarray:
    # Explicitly broadcast to prevent shape errors
    scale_matrix = np.broadcast_to(np.subtract(scales[0, :], scales[1, :]), matrix.shape)
    return np.add(np.multiply(matrix, np.subtract(scales[0, :], scales[1, :])), scales[0, :])


# Transform a numpy array to [-1, 1]
def transform_minus_one_to_one(input: np.ndarray) -> np.ndarray:
    return scale_matrix_columns(input, -1, 1)
