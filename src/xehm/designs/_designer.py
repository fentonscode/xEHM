#
# A designer takes a set of variables and creates design samples
#

from typing import List
from .._variables import Variable

__all__ = ["Designer"]


class Designer:
    def __init__(self):
        if type(self) is Designer:
            raise Exception("Emulator is an abstract base, inherit and define your own")

    def design(self, v_list: List[Variable], n_points: int):
        raise NotImplementedError("Custom designers must provide a design method")