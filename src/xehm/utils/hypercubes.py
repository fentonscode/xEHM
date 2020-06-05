import numpy as np


# Draw a uniform sample from an n-dimensional hypercube
# limits: min/max limits for each variable - Matrix[dims, 2], column 1 = min, column 2 - max
# NOTE: np.random.uniform adds extra container fluff - squeeze the output
def uniform_box(limits: np.ndarray) -> np.ndarray:
    dims: int = limits.shape[0]
    return np.array([np.random.uniform(low=limits[i, 0], high=limits[i, 1], size=1)
                     for i in range(dims)]).squeeze()


# There might be a better way to do this without list comprehension
# x should be squeezed to eliminate spurious dimensions
def point_in_box(x: np.ndarray, box_limits: np.ndarray) -> bool:
    return np.all([box_limits[i, 0] <= x[i] <= box_limits[i, 1] for i, _ in enumerate(x)])


# Generates a hypercube of samples based on the axes and size supplied
# Axes: List of tuples of (start, stop) pairs - e.g. [(0.0, 1.0), (-1.0, 1.0)]
# Size: number of samples (same across axes)
def hypercube(axes, size):
    ndims = len(axes)
    rev_axes = axes[::-1]
    L = [np.linspace(rev_axes[i][0], rev_axes[i][1], size) for i in range(ndims)]
    return np.flip(np.hstack((np.meshgrid(*L))).swapaxes(0, 1).reshape(ndims, -1).T, axis=1)
