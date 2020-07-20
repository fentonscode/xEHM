#
# Some explicit Bayesian schemes
#

from typing import List, Union

__all__ = ["bernoulli_beta"]


# Return the probability of a non-zero result given a train of samples over a beta prior
def bernoulli_beta(x: List[Union[bool, int]], prior_a: float = 1.0, prior_b: float = 1.0):
    binary_data: List[int] = [1 if s >= 1.0 else 0 for s in x]
    n = len(x)
    sx = sum(binary_data)
    post_a = prior_a + sx
    post_b = prior_b + n - sx
    mean = post_a / (post_a + post_b)
    mode = (post_a - 1.0) / (post_a + post_b - 2.0)
    variance = (post_a * post_b) / (((post_a + post_b) ** 2) * (post_a + post_b + 1.0))
    return mean, mode, variance, post_a, post_b


