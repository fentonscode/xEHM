import numpy as np
from .samplers import Sampler
from .._distributions import Distribution


# Rejection sampler algorithm:
#
# Implemented as:
#   - Class object (for higher-order structures which need to understand what a sampler is)
#   - Callable function (for general use)
#
# NOTES:
#   - Both implementations call the _reject_fast() method, main algorithm is in there
#
class RejectionSampler(Sampler):
    def __init__(self, pdf: Distribution):
        super().__init__(pdf)
        self.ident = "Rejection Sampler"

    # Run an N-Dimensional rejection sampler
    def run(self, num_samples: int, params):
        if num_samples < 1:
            raise ValueError("num_samples must be greater than 0")
        upper_boundary = float(params[0])
        if upper_boundary < 0.0:
            raise ValueError("Rejection boundary must be positive")
        self.last_run = np.asarray(_reject_fast(self.distribution.probability, self.distribution.num_dimensions,
                                                num_samples, self.distribution.support_limits[:, 0],
                                                self.distribution.support_limits[:, 1],
                                                upper_boundary))
        self.acceptance = self.last_run.shape[0] / num_samples
        return self


# Rejection sampler function (use this if you don't need class objects)
def rejection_sampler_nd(p_dist, num_samples: int, min_x, max_x, scale: float) -> np.ndarray:
    # Argument checks
    if not callable(p_dist):
        raise TypeError("p_dist must be a callable probability density function")
    if not type(num_samples) is int:
        raise TypeError("num_samples must be a positive integer")
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not type(min_x) in (list, tuple):
        raise TypeError("min_x must be a list or tuple")
    if not all(type(x) in (int, float) for x in min_x):
        raise TypeError("min_x sequence must be all numbers")
    if not type(max_x) in (list, tuple):
        raise TypeError("max_x must be a list or tuple")
    if not all(type(x) in (int, float) for x in max_x):
        raise TypeError("max_x sequence must be all numbers")
    if not type(scale) in (int, float):
        raise TypeError("scale must be a number")
    scalef = float(scale)
    if scalef < 0.0:
        raise ValueError("scale must be positive")
    if len(min_x) != len(max_x):
        raise ValueError("min_x and max_x are not the same size")

    return _reject_fast(p_dist, len(min_x), num_samples, min_x, max_x, scale)


# Rejection sampler (_reject_fast)
# Designed to be called internally only, does no error checking at all
#
# Parameters:
#   - pdf_function: Must be a python function which operates over numpy ndarrays and can be vectorised
#   - n_dims: number of dimensions, potentially prevents expensive length calculations if the shape isn't available
#   - min_x: iterable list (something that can be indexed with []) of minimum supports per dimension
#   - max_x: iterable list (something that can be indexed with []) of maximum supports per dimension
#   - scale: Upper boundary of the uniform distribution which bounds the pdf_function completely
#
# Outputs
#   - Accepted samples in a numpy.adarray
#
# NOTES:
#   - This could be even faster with some better usage of numpy vector functions
#   - min_x and max_x are separated for flexibility to call with lists instead
#
def _reject_fast(pdf_function, n_dims: int, n_samples: int, min_x, max_x, scale: float) -> np.ndarray:
    # NOTE: As typically num_samples >> dims, create the largest vectors in the smallest loop
    p_alloc = np.empty((n_dims, n_samples), dtype=float)
    p_alloc = np.asarray([np.random.uniform(min_x[i], max_x[i], size=n_samples) for i, _ in enumerate(p_alloc)])
    x_values = p_alloc.transpose().squeeze()
    y_values = np.random.uniform(0.0, scale, n_samples)
    p_values = pdf_function(x_values)
    return np.asarray([x for i, x in enumerate(x_values) if not y_values[i] > p_values[i]])
