#
# The default designer creates random draws from the support of each variable independently
#

from typing import List
import numpy as np
from ._designer import Designer
from .._variables import Variable
from ..utils import uniform_box

__all__ = ["DefaultDesigner"]


class DefaultDesigner(Designer):
    def __init__(self):
        super().__init__()

    def design(self, v_list: List[Variable], n_points: int):
        mins = [var.min_support for var in v_list]
        maxs = [var.max_support for var in v_list]
        return uniform_box(np.asarray([[mins], [maxs]]), n_points)