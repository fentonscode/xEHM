from .figure_controls import HGraph, vertical_split_percent
import numpy as np

__all__ = ["plot_1d_nroy"]

#
# NROY Space plotting
# -------------------
#
# NROY Space is an abstract representation of the parameter space defined by finite collections of samples
# Plotting this space is a transformation from NROY space to plot space, which will need either
# up or downsampling depending on the resolutions of both spaces
#


def samples_to_plot_space_1d(samples: np.ndarray, plot_resolution):
    pass


def generate_plot_space(nroy_space: np.ndarray, plot_resolution):

    # NROY space is an array with as many dimensions as there are independent inputs
    # The length of each dimension is the resolution of each axis
    resolutions = nroy_space.shape
    plot_space = np.zeros([plot_resolution] * len(resolutions))


# implausibility_to_rgb
#
# This creates a simple RGB triple based on the implausibility of the current NROY space
# For best compatibility with publications, the colour mapping is as follows:
#   - I(x) < 0: Red
#   - 0 < I(x) < 3: Yellow to Red linear blend
#   - I(x) > 3: Red
#
# TODO: Convert to a matplotlib colour map at some point
# TODO: Make the cut off and colours parameters?
def implausibility_to_rgb(imp: float) -> (float, float ,float):
    if imp < 0.0 or imp > 3.0:
        return 1.0, 0.0, 0.0
    else:
        return 1.0, (3.0 - imp) / 3.0, 0.0


# wave_to_rgb
#
# This creates a simple RGB triple based on the maximum non-implausible wave status of the
# current NROY space. Usage appears to vary by publication, but the following is proposed:
#   - 0 < w(x) < W: Black to Blue linear discrete blend
# Where W is the maximum number of waves, 0 is unexplored space and w(x) is the wave at which
# the NROY region is ruled out
#
def wave_to_rgb(wave: int, max_waves: int) -> (float, float, float):
    return 0.0, float(wave) / float(max_waves), 0.0


def make_colour_map():
    # NROY Space colours
    # Red = nothing
    # Yellow = zero implausibility
    colours = [(1, 1, 0), (1, 0, 0)]
    breaks = [0.0, 3.0]


# Helper structure for 1D nroy slices
class nroy_1d_chunk:
    def __init__(self):
        self.x_start: float = 0.0
        self.x_stop: float = 0.0
        self.implausibility: float = 0.0
        self.sample_count: int = 0

    # Accumulate all implausibility
    def accumulate(self, implausibility_value):
        self.sample_count += 1
        self.implausibility += implausibility_value

    # Store an average of the implausibility in the region
    def average(self, implausibility_value):
        total = self.implausibility * self.sample_count
        total += implausibility_value
        self.sample_count += 1
        self.implausibility = total / self.sample_count

    # Store the maximum implausibility in the region
    def maximum(self, implausibility_value):
        if implausibility_value > self.implausibility:
            self.implausibility = implausibility_value
        self.sample_count += 1


def plot_1d_nroy(min_x: float, max_x: float, x_locs: np.ndarray, implausibility: np.ndarray, resolution: int, cut_off: float):
    if resolution < 2:
        raise ValueError("Resolution too low")
    if np.any(x_locs < min_x) or np.any(x_locs > max_x):
        raise ValueError("Samples outside of support")

    boundaries = np.linspace(min_x, max_x, resolution + 1)
    x_range = max_x - min_x
    chunks = []
    for i in range(resolution):
        chunk_def = nroy_1d_chunk()
        chunk_def.x_start = boundaries[i]
        chunk_def.x_stop = boundaries[i + 1]
        chunks.append(chunk_def)

    for x, i in zip(x_locs, implausibility):
        chunk_loc = int(((x[0] - min_x) / x_range) * resolution)
        chunks[chunk_loc].average(i[0])

    graph = vertical_split_percent(0.1)
    non_imp = x_locs[implausibility <= cut_off]
    graph.axes[0].hist(non_imp, bins=resolution, density=True)

    for c in chunks:
        if c.sample_count > 0:
            colour = implausibility_to_rgb(c.implausibility)
        else:
            colour = implausibility_to_rgb(-1.0)
        graph.axes[1].fill_between(x=[c.x_start, c.x_stop], y1=0.0, y2=1.0, color=colour)

    # Set axis limits
    graph.axes[0].set_xlim([min_x, max_x])
    graph.axes[1].set_xlim([min_x, max_x])
    graph.axes[1].axes.yaxis.set_visible(False)
    graph.axes[1].set(xlabel="x")

    graph.plot()


def plot_nroy(n_dims: int, min_x, max_x, samples, resolution):
    if n_dims < 1:
        raise ValueError("Invalid dimensions")
    if n_dims == 1:
        plot_1d_nroy(min_x, max_x, samples, resolution)

    num_plots = n_dims ** 2
    graph = HGraph().set_dimensions(n_dims, n_dims)