#
# The default designer creates random draws from the support of each variable independently
#

from typing import Tuple, Union
import numpy as np
from ._designer import Designer
from .._variables import Variable
from ..utils import uniform_box

__all__ = ["DefaultDesigner", "default_designer"]


class DefaultDesigner(Designer):
    def __init__(self):
        super().__init__()

    def design(self, v_list: Tuple[Variable], n_points: int):
        return default_designer(v_list, n_points)


#
# The default designer calls randomly from the support of each input variable
#
def default_designer(v_list: Union[Variable, Tuple[Variable]], n_points: int):
    if not isinstance(Variable, tuple):
        return uniform_box(np.asarray([[v_list.min_support], [v_list.max_support]]), n_points)
    mins = [var.min_support for var in v_list]
    maxs = [var.max_support for var in v_list]
    return uniform_box(np.asarray([[mins], [maxs]]), n_points)