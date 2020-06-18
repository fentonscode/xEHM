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

    #
    # propose: Generates n samples mapped from a list of input variables
    #
    def propose(self, v_list: List[Variable], n_points: int):
        raise NotImplementedError("Custom designers must provide a propose method")

    #
    # select: Choose n samples from a given set
    #
    def select(self, s_list, n_points: int):
        raise NotImplementedError("Custom designers must provide a select method")
