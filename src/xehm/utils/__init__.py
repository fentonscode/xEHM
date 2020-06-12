from .console import *
from .hypercubes import *
from .numpy_tools import *
from .serialisation import *
from .spline_fits import *


# Mean square error
def mse(a, b):
    return np.square(a - b).mean()


# Maximum square error
def mmse(a, b):
    return np.square(a - b).max()
