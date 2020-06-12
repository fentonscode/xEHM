from typing import List, Union
import numpy as np
from ._variables import Variable
from .emulators import Emulator
from .utils.hypercubes import uniform_box

__all__ = ["HistoryMatching"]


# Overall history match object to use from within Python (rather than the command line)
class HistoryMatching:
    def __init__(self):
        self.build_emulator = None
        self.out_space = None
        self.input_variables: List[Variable] = []
        self.nroy_resolution = 100
        self.space_indices: List[List[int]] = []
        self.emulators: List[Emulator] = []
        self.current_wave = 0
        self.n_dimensions = 0
        self.out_samples: Union[np.ndarray, None] = None
        self.simulator = None

    def load(self, f_name: str):
        pass

    def save(self, f_name: str):
        pass

    # TODO: Check for copy problems here
    def set_input_variables(self, var_list: List[Variable]):
        if len(var_list) == 0:
            raise ValueError("Variable list is empty")
        self.input_variables = var_list
        self.n_dimensions = len(var_list)

    # TODO: Make simulator wrappers
    def set_simulator(self, sim_function):
        if not callable(sim_function):
            raise ValueError("sim_function must be callable")
        self.simulator = sim_function

    # This resets the history matching object back to the start
    def reset(self):
        self.space_indices: List[List[int]] = []
        self.emulators: List[Emulator] = []
        self.current_wave = 0
        self.out_samples = None

    def run(self, n_design_points):

        # The first stage of history matching is a bit special, it requires an explicit design
        if self.current_wave == 0:

            # Design phase
            # TODO: Make this a plugin as well
            mins = [var.min_support for var in self.input_variables]
            maxs = [var.max_support for var in self.input_variables]

            initial_design_points = uniform_box(np.asarray([[mins], [maxs]]), n_design_points)
            initial_design_outputs = self.simulator(initial_design_points)