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


# Below are some junk functions from an old project

# Helper function for plotting x-y pairs
def plot_xy(xy_tuples):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    acceptx = [n[0] for n in xy_tuples]
    accepty = [n[1] for n in xy_tuples]
    ax.scatter(acceptx, accepty, c='g', marker='+')
    fig.show()


def plot_2d(x, y):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    acceptx = [n[0] for n in x]
    accepty = [n[1] for n in x]
    plot = ax.scatter(acceptx, accepty, c=y, cmap="RdYlBu", marker='+')
    fig.colorbar(plot, ax=ax)
    fig.show()


def plot_2d_samples(samples: np.ndarray, implausibility: np.ndarray = None, i_cut_off: float = 3.0):
    if samples.shape[1] != 2:
        raise ValueError("Samples must be a N x 2 array")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    # Bit of a hack-y way to do it, should look into colour mapping instead?
    if implausibility is not None:
        mask = implausibility[:, 0] <= i_cut_off
        non_imp = samples[mask]
        imp = samples[~mask]
        x1 = non_imp[:, 0]
        y1 = non_imp[:, 1]
        x2 = imp[:, 0]
        y2 = imp[:, 1]
        ax.scatter(x1, y1, marker='+', color="g")
        ax.scatter(x2, y2, marker='+', color="r")
    else:
        x = samples[:, 0]
        y = samples[:, 1]
        ax.scatter(x, y, marker='+', color="b")
    # FIXME: sort this later
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.show()