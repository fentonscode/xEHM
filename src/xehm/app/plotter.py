# NROY Space plotter to go here

import matplotlib.pyplot as plt
from xehm.sampling.samplers import Sampler


# Sampler Plotting code:

# Projects a 2D sampler into a set of plot axes
def __make_2d_axes(s: Sampler, axes):
    axes.scatter(s.last_run[:, 0], s.last_run[:, 1], marker='+')
    axes.set(title=s.ident + ": acceptance = {:.3f}".format(s.acceptance),
             xlabel="$x_1$", ylabel="$x_2$")
    axes.set_xlim([s.distribution.support_limits[0, 0], s.distribution.support_limits[0, 1]])
    axes.set_ylim([s.distribution.support_limits[1, 0], s.distribution.support_limits[1, 1]])


# Plots a 2D sampler output in a single graph
def plot_2d_sampler(s: Sampler):
    if s.last_run is None:
        print("Attempt to plot sampler with no run data. Perform a run first")
        return
    if s.distribution is None or s.distribution.num_dimensions != 2:
        print("Plotting is only supported for 2D distributions")

    plt.style.use('default')
    frame = plt.figure(figsize=(7, 7))
    axes = frame.add_subplot(111)
    __make_2d_axes(s, axes)
    frame.show()


# Plots a pair of sampler outputs side-by-side for comparison
def plot_compare_2d_sampler(s1: Sampler, s2: Sampler):
    if s1.last_run is None or s2.last_run is None:
        print("Attempt to plot sampler with no run data. Perform a run first")
        return
    if (s1.distribution is None or s1.distribution.num_dimensions != 2 or s2.distribution is None
            or s2.distribution.num_dimensions != 2):
        print("Plotting is only supported for 2D distributions")
    plt.style.use('default')
    frame = plt.figure(figsize=(14, 7))
    left = frame.add_subplot(121)
    right = frame.add_subplot(122)
    __make_2d_axes(s1, left)
    __make_2d_axes(s2, right)
    frame.show()
