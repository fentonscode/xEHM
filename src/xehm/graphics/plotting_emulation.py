#
# Plot things to do with emulators
#

from ..emulators import Emulator
import matplotlib.pyplot as plt
import numpy as np


def plot_emulator_design_points(axes, x, y):
    axes.plot(x, y, 'kx')
    return axes


# Plot a SISO emulator trace, all numerical inputs are column vectors
def plot_emulator_single_output(axes, x: np.ndarray, mean: np.ndarray, variance: np.ndarray):
    delta = 2.0 * np.sqrt(variance[:, 0])
    upper = mean[:, 0] + delta
    lower = mean[:, 0] - delta
    axes.plot(x[:, 0], mean[:, 0], color='b')
    axes.fill_between(x[:, 0], lower, upper, color="C0", alpha=0.2)
    return axes


def plot_emulator_for_wave(axes, wave_num, z_match, z_variance, x_min, x_max, x_list, mean_list, variance_list):
    z_delta = 2.0 * np.sqrt(z_variance)
    z_upper = z_match + z_delta
    z_lower = z_match - z_delta
    for x, mean, variance in zip(x_list, mean_list, variance_list):
        axes = plot_emulator_single_output(axes, x, mean, variance)
    axes.fill_between(x=[x_min, x_max], y1=z_lower, y2=z_upper, color='green', alpha=0.25)
    axes.hlines(y=z_match, xmin=x_min, xmax=x_max, color='k', linestyle='dashed')
    axes.set(xlabel='x', ylabel='z', title=f'Emulator output - wave {wave_num}')
    axes.grid(linestyle='--', color='grey', alpha=0.5)
    return axes


#def plot_cross_validation(emulator_x, emulator_y, cv_means, cv_variances):
#    fig, axes = make_single_figure_axes()
#    axes.plot(emulator_x, emulator_y, 'kx')
#    fig.show()