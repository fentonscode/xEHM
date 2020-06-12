import numpy as np
from scipy.stats import multivariate_normal

__all__ = ["Distribution", "Tree2D", "Proposal", "UniformStepProposal"]


# Distributions define the implausibility functions that are used to power IDEMC
# These hold
#   - Number of dimensions: (int) used throughout samplers
#   - Support limits: (List[float]) two-column array of low/high limits per dimension (dimension per row)
#
# Each base distribution must define a probability density function that takes a location and parameter list

class Distribution:
    def __init__(self, dims: int, limits: np.ndarray = None):
        # Default states
        self.num_dimensions = -1
        self.support_limits = None
        self.params = None

        # Type / parameter checks
        if type(self) is Distribution:
            raise Exception("Distribution is an abstract base, inherit and define your own")
        if dims is None or dims < 1:
            raise ValueError("dims must be > 0")
        self.num_dimensions = dims
        if limits is None:
            self.support_limits = np.broadcast_to(np.asarray([-np.inf, np.inf]), (dims, 2))
        elif limits.shape[0] != dims or limits.shape[1] != 2:
            raise ValueError("limits should be a [dimensions x 2] shape matrix")
        else:
            self.support_limits = limits

    def probability(self, x):
        raise NotImplementedError("Custom distributions must define a probability density function")

    def set_params(self, params):
        raise NotImplementedError("Custom distributions must define a function to set latent paramters")

    def temper(self, x, t):
        return np.exp(np.divide(np.negative(self.probability(x)), t))


# Proposal distributions are a template for MCMC proposals
# This is a thin wrapper that just enforces the right methods being available
class Proposal(Distribution):

    def __init__(self, dims: int, limits: np.ndarray = None):
        super().__init__(dims, limits)

    def random_draw(self, x) -> np.ndarray:
        raise NotImplementedError("Custom proposals must define a random draw function")

    def transition_kernel(self, new: np.ndarray, old: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Custom proposals must define a transition kernel")


class Williamson2D(Distribution):

    def __init__(self):
        limit = np.broadcast_to(np.array([-3.0, 7.0]), (2, 2))
        super().__init__(2, limit)
        v1 = np.asarray([[0.4, 0.0], [0, 0.008]])
        v2 = np.asarray([[0.08, 0.186], [0.186, 0.48]])
        self.sigma1 = np.linalg.inv(v1)
        self.sigma2 = np.linalg.inv(v2)
        self.m1 = np.asarray([1.6, 1.7])
        self.m2 = np.asarray([1.3, 1.3])
        self.volume = 1.0 / 100

    def implausibility(self, x):
        k1 = np.subtract(x, self.m1)
        k2 = np.subtract(x, self.m2)
        s_dev1 = np.sqrt(np.matmul(np.transpose(k1), np.matmul(self.sigma1, k1)))
        s_dev2 = np.sqrt(np.matmul(np.transpose(k2), np.matmul(self.sigma2, k2)))
        return np.minimum(s_dev1, s_dev2)

    # For IDEMC / mcmc comparisons the probability is modified to account for implausibility
    def probability(self, x: np.ndarray) -> np.ndarray:
        # Implausibility is not quite vectorised yet
        a = x.reshape(-1, self.num_dimensions)
        imp_list = np.apply_along_axis(self.implausibility, 1, a)
        return np.where(imp_list <= 3.0, self.volume, 0.0)

    def set_params(self, params):
        pass


# Williamson 2013 10 dimensional distribution
# Convert the globals from R into a class to stop recalculating constants every iteration
class Tiny10dDistribution(Distribution):

    def __init__(self):
        limit = np.array([-3.0, 7.0])
        super().__init__(10, np.repeat(limit[np.newaxis, :], 10, 0))
        vs1 = np.array([0.1, 0.0125, 0.025, 0.04, 0.01])
        vs1 = np.multiply(np.r_[vs1, vs1], 0.5838968 ** 2)
        vs2 = np.array([0.025, 0.1, 0.01, 0.01, 0.05])
        vs2 = np.multiply(np.r_[vs2, vs2], 0.5838968 ** 2)
        c1 = 0.85
        v1 = np.diag(vs1)
        v2 = np.diag(vs2)
        k = 0
        for i in range(self.num_dimensions):
            for j in range(i + 1, self.num_dimensions):
                v1[i, j] = c1 * np.sqrt(vs1[i]) * np.sqrt(vs1[j])
                v1[j, i] = v1[i, j]
                v2[i, j] = c1 * np.sqrt(vs2[i]) * np.sqrt(vs2[j])
                v2[j, i] = v2[i, j]
                k += 1
        self.sigma1 = np.linalg.inv(v1)
        self.sigma2 = np.linalg.inv(v2)
        self.m1 = np.repeat(1.0, self.num_dimensions)
        self.m2 = np.array([4, 3, 3, 4, 3, 4, 4, 4, 2, 2])
        self.volume = 1.0 / (10 ** 10)

    # Calculate implausibility - reuses the constants initialised earlier
    # If bigger than max_i - then do something??
    def implausibility(self, x):
        k1 = np.subtract(x, self.m1)
        k2 = np.subtract(x, self.m2)
        s_dev1 = np.sqrt(np.matmul(np.transpose(k1), np.matmul(self.sigma1, k1)))
        s_dev2 = np.sqrt(np.matmul(np.transpose(k2), np.matmul(self.sigma2, k2)))
        return np.minimum(s_dev1, s_dev2)

    # For IDEMC / mcmc comparisons the probability is modified to account for implausibility
    def probability(self, x: np.ndarray) -> np.ndarray:
        # Implausibility is not quite vectorised yet
        a = x.reshape(-1, self.num_dimensions)
        imp_list = np.apply_along_axis(self.implausibility, 1, a)
        return np.where(imp_list <= 3.0, self.volume, 0.0)


class Tree2D(Distribution):
    def __init__(self):
        super().__init__(2, np.asarray([0.0, 1.0, 0.0, 1.0]).reshape(2, 2))
        self.means = np.asarray([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
        self.covariance = np.asarray([0.0515 ** 2, 0, 0, 0.0515 ** 2]).reshape(2, 2)

    # OLD CODE FROM mlab
    @staticmethod
    def __bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0):
        Xmu = X - mux
        Ymu = Y - muy
        rho = sigmaxy / (sigmax * sigmay)
        z = Xmu ** 2 / sigmax ** 2 + Ymu ** 2 / sigmay ** 2 - 2 * rho * Xmu * Ymu / (sigmax * sigmay)
        denom = 2 * np.pi * sigmax * sigmay * np.sqrt(1 - rho ** 2)
        return np.exp(-z / (2 * (1 - rho ** 2))) / denom

    def probability(self, x):
        a = x.reshape(-1, 2)
        p2: float = 0.0
        for m in self.means:
            p2 += multivariate_normal.pdf(x, mean=m, cov=self.covariance)
        # p = multivariate_normal.pdf(x, mean=self.means[0], cov=self.covariance)
        x1 = a[:, 0]
        x2 = a[:, 1]
        in_box = lambda k: np.all(
            [self.support_limits[i, 0] <= k[i] <= self.support_limits[i, 1] for i, _ in enumerate(k)])
        p = np.multiply(0.25, self.__bivariate_normal(x1, x2, 0.0515, 0.0515, 0.25, 0.25, 0.0))
        p += np.multiply(0.25, self.__bivariate_normal(x1, x2, 0.0515, 0.0515, 0.25, 0.75, 0.0))
        p += np.multiply(0.25, self.__bivariate_normal(x1, x2, 0.0515, 0.0515, 0.75, 0.25, 0.0))
        p += np.multiply(0.25, self.__bivariate_normal(x1, x2, 0.0515, 0.0515, 0.75, 0.75, 0.0))
        return np.where(np.apply_along_axis(in_box, 1, a), p, 0.0)


# Uniform distribution: used as a default proposal
# This is generalised for n-dimensions and n support limits (one per dimension)
#
# NOTES:
#   - All proposal vectors are drawn between the limits for each component in a uniform manner
#   - As this is symmetric, the transition kernel returns a constant probability for all inputs
#   - The volume of the space is the product of all the widths of each support limit
# TODO: Fix the zero centreing problem
class UniformStepProposal(Proposal):

    # Constructor - sets up the proposal for future use
    # Parameters
    #   - ndims: an integer specifying the number of dimensions
    #   - limits: a numpy.ndarray of size [ndims x 2] containing [low, high] for each dimension
    #
    # NOTES:
    #   - If ndims is missing, negative or zero __init__ will fail and raise an exception
    #   - If limits is missing it will default to [-1, 1] for all dimensions

    def __init__(self, ndims: int, limits: np.ndarray = None):
        if ndims is None or ndims < 1:
            raise ValueError("Uniform proposal requires dimensions")
        if limits is None:
            limit = np.broadcast_to(np.array([-1.0, 1.0]), (ndims, 2))
        else:
            limit = limits.reshape(-1, 2)
        super().__init__(ndims, limit)
        self.volume = 1.0 / np.prod(np.subtract(limit[:, 1], limit[:, 0]))

    # Draw multi-dimensional random steps across each set of limits
    def random_draw(self, x) -> np.ndarray:
        jump = np.asarray([np.random.uniform(low=self.support_limits[i, 0], high=self.support_limits[i, 1],
                                             size=1) for i in range(self.num_dimensions)]).transpose().squeeze()
        return np.add(x, jump)

    # In this (symmetric) walker, distributions are constant, but check for going out of bounds
    # NOTE: We check the jump as the support on this is finite (and should be smaller than the target density)
    def transition_kernel(self, new: np.ndarray, old: np.ndarray) -> np.ndarray:
        jump = np.subtract(new, old)
        return np.where(np.all([self.support_limits[i, 0] <= jump[i] <= self.support_limits[i, 1]
                                for i, _ in enumerate(jump)]), self.volume, 0.0)


# Normal random walker with the same variance on each dimension
class NormalStepProposal(Proposal):

    def __init__(self, ndims: int, scale: float):
        if ndims is None or ndims < 1:
            raise ValueError("Uniform proposal requires dimensions")
        super().__init__(ndims)
        self.covariance = np.identity(ndims) * scale

    # MVN proposals are zero mean for random draws
    def random_draw(self, x) -> np.ndarray:
        return np.random.multivariate_normal(mean=x, cov=self.covariance, size=1).squeeze()

    # In this (symmetric) walker, the current location is the mean for the transition density
    def transition_kernel(self, new: np.ndarray, old: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(new, mean=old, cov=self.covariance)


# Here are a few convenient methods for some pre-set proposals for typical analyses

# Convenience class to construct random walkers
def make_random_walker(ndims: int, step_size: float, uniform=True):
    if uniform:
        limits = np.broadcast_to(np.asarray([-step_size, step_size]), (ndims, 2))
        return UniformStepProposal(ndims, limits)
    else:
        return NormalStepProposal(ndims, step_size)
