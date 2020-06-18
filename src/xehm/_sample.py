#
# This is here because it makes tracking the information about the space a lot easier
#

from typing import Union
import numpy as np


__all__ = ["Sample", "SampleSet", "SISOSampleSet"]


class Sample:
    def __init__(self, loc: Union[np.ndarray, None] = None, imp: Union[np.ndarray, None] = None,
                 em: Union[np.ndarray, None] = None):
        self.location: Union[np.ndarray, None] = loc
        self.implausibility: Union[np.ndarray, None] = imp
        self.emulated: Union[np.ndarray, None] = em


#
# Sample sets are collections of samples (set membership works) that operate inside a numpy array for speed
#
class SampleSet:
    def __init__(self, in_dims: int, out_dims: int, n_points: int):
        if in_dims < 1 or out_dims < 1:
            raise ValueError("Invalid dimensions")
        total_width = in_dims + out_dims + 1
        self._input_dimensions = in_dims
        self._output_dimensions = out_dims
        self._data = np.zeros((n_points, total_width))

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        if isinstance(value, tuple):
            self.write_components(key, value[0], value[1], value[2])
        else:
            self.write_sample(key, value)

    def write_sample(self, index, value):
        self._data[index] = value

    def write_components(self, index, location, emulation, implausibility):
        p1 = self._input_dimensions + self._output_dimensions
        self._data[index, 0:self._input_dimensions] = location
        self._data[index, self._input_dimensions:p1] = emulation
        self._data[index, p1:] = implausibility

    def __len__(self):
        return self._data.shape[0]


# Single input, single output set for 1d history matching
class SISOSampleSet:
    def __init__(self, n_points):
        self._data = np.zeros((n_points, 4))

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        if isinstance(value, tuple):
            self.write_components(key, value[0], value[1], value[2], value[3])
        else:
            self.write_sample(key, value)

    def __len__(self):
        return self._data.shape[0]

    def resize(self, n_points):
        self._data = np.zeros((n_points, 4))

    def write_sample(self, index, value):
        self._data[index] = value

    def write_components(self, index, location, emulation, implausibility):
        self._data[index, 0] = location
        self._data[index, 1] = emulation[0]
        self._data[index, 2] = emulation[1]
        self._data[index, 3] = implausibility

    def sort_by_location(self):
        self._data = self._data[self._data[:, 0].argsort()]
