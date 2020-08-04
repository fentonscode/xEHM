#
# The default designer creates random draws from the support of each variable independently
#

from typing import Tuple, Union
import numpy as np
from ._designer import Designer
from .._variables import Variable
from ..utils import uniform_box
from ..utils.numpy_tools import random_sample

__all__ = ["DefaultDesigner", "default_designer", "default_selector"]


class DefaultDesigner(Designer):
    def __init__(self):
        super().__init__()

    def propose(self, v_list: Tuple[Variable], n_points: int):
        return default_designer(v_list, n_points)


#
# The default designer calls randomly from the support of each input variable
#
def default_designer(variables: Union[Variable, Tuple[Variable]], points: int):
    if not isinstance(variables, tuple):
        return uniform_box(np.asarray([[variables.min_support], [variables.max_support]]), points)
    mins = [var.min_support for var in variables]
    maxs = [var.max_support for var in variables]
    return uniform_box(np.asarray([mins, maxs]), points)


def default_selector(s_list: np.ndarray, n_points: int):
    # Short-circuit if all samples are selected
    if n_points >= s_list.shape[0]:
        return s_list

    # Randomly select from s_list
    return random_sample(s_list, n_points)


#
# selector_expanding: choose from a candidate set
#
# NOTES:
#   - If n_points equals the number of samples, then the samples are returned
#   - If n_points > the number of samples, then return equal distribution + random selection of the remainder
#   - If n_points < the number of samples, then return a random selection
#
def selector_expanding(s_list: np.ndarray, n_points: int):

    # Short-circuit if all samples are selected
    if n_points == s_list.shape[0]:
        return s_list

    output: np.ndarray = np.zeros((n_points, s_list.shape[1]))
    whole_copies: int = n_points // s_list.shape[0]

    # If we want more points than there are available, copy across the entire set as a duplication
    for k in range(whole_copies):
        start: int = k * n_points
        end: int = (k + 1) * n_points
        output[start:end] = s_list

    # FIXME: Is this the right divisor order?
    remaining = n_points % s_list.shape[0]

