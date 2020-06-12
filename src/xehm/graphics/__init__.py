#
# Sub-package for plotting stuff
#

import matplotlib.pyplot as plt
from .plotting_emulation import *
from .figure_controls import *
from .plotting_nroy import *


def make_single_figure_axes():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    return fig, ax