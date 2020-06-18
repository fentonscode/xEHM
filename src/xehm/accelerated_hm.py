#
# Accelerated History Matching for low dimensional problems
#

#
# One dimensional cases
#
# In 1D history matching is likely extremely inefficient compared to other methods, but is valid
# Regions and space splits in 1D are analytically trivial, so cascading samples through the entire
# emulator hierarchy can be avoided for a simpler and faster implementation
#

from .space_descriptor import RegionOneDimensional
from .clustering import XMeans
from .emulators import GaussianProcess
from .designs import default_designer, default_selector
from .diagnostics import LeaveOneOut
from .graphics import plot_1d_nroy, plot_emulator_for_wave, plot_emulator_design_points
from .graphics import HGraph
from ._sample import SISOSampleSet
from ._variables import Variable
from .utils import implausibility
from typing import Union, List
from sklearn.cluster import KMeans
import numpy as np

__all__ = ["HistoryMatching1D"]


# Parameters
#   - Minimum support
#   - Maximum support
#   - Calibration mean
#   - Calibration variance
#   - Simulator
#   - Emulator (or use default)
#   - Designer (or use default)

class HistoryMatching1D():
    def __init__(self, min_x, max_x, z_mean, z_variance, emulator_model=GaussianProcess):
        self._variable = Variable("x", min_x, max_x)
        self._target_mean = z_mean
        self._target_variance = z_variance
        self._space = RegionOneDimensional(self._variable.min_support, self._variable.max_support)
        self._sim_function = None
        self._num_waves = 0
        self._emulators = []
        self._current_samples: Union[SISOSampleSet, None] = None
        self._emulator_budget = 100
        self._simulator_budget = 10
        self.sim_design_points = []
        self.sim_runs = []

        # Analysis components
        self._emulator_model = emulator_model
        self._designer = default_designer
        self._diagnostic = LeaveOneOut.cross_validate

        # Outputs / data for plotting
        self._samples = None

    def set_simulator(self, simulator):
        self._sim_function = simulator

    def set_budgets(self, sim_max, em_max):
        self._simulator_budget = sim_max
        self._emulator_budget = em_max

    #
    # Initialise: Setup the first components for successive history matching waves
    #
    # n_points: This sets the simulator budget for the process.
    # i_cut_off: implausibility cut-off for the initial samples
    #
    # Notes:
    #   - If the simulator has not been defined / attached then this will fail
    #
    def initialise(self, n_points: Union[int, None] = None):

        # If n_points is specified then update the simulator budget
        if n_points is not None:
            self._simulator_budget = n_points
        if self._sim_function is None:
            raise ValueError("No simulator defined for this history matching process")

        # Create the initial design points using the simulator
        initial_design = self._designer(self._variable, self._simulator_budget)
        initial_runs = self._sim_function(initial_design)

        # Build and validate the emulator
        emulator = self._emulator_model()
        valid = self._diagnostic(emulator, initial_design, initial_runs)
        if not valid:
            print(f"Emulation failed in initial wave")
            # What do we do here?
        emulator.train(initial_design, initial_runs)
        print("Constructed initial emulator")

        # Generate emulation samples
        emulator_x = self._designer(self._variable, self._emulator_budget)
        means, variances = emulator.evaluate(emulator_x)
        emulator_i = implausibility(self._target_mean, means, self._target_variance, variances)

        # Don't select out final samples to allow full space plotting
        self._samples = np.zeros((self._emulator_budget, 4))
        self._samples[:, 0] = emulator_x[:, 0]
        self._samples[:, 1] = means[:, 0]
        self._samples[:, 2] = variances[:, 0]
        self._samples[:, 3] = emulator_i[:, 0]
        self._emulators.append(emulator)

    def run_wave(self, i_cut_off=3.0):

        # Check the implausibility space and filter out 'good' samples
        self._samples = self._samples[self._samples[:, 3] <= i_cut_off]

        # Numpy ruins 1D slices, so weld the dimension back on
        locations = self._samples[:, 0].reshape(-1, 1)

        # Each wave starts by inspecting the current available samples and discovering structure
        # np.newaxis stops numpy chopping off the dimensions
        cl = XMeans().assign(locations, None)
        print(f"Recommending {len(cl)} clusters")

        # TODO: Do we need to call KMeans again??
        clusters = KMeans(n_clusters=cl.n_groups).fit(locations)

        # Split space using cluster centres
        splits = []
        locs = sorted(clusters.cluster_centers_)
        if cl.n_groups > 1:
            splits = [0.5 * (locs[i + 1] + locs[i]) for i in range(cl.n_groups - 1)]
        for split in splits:
            self._space.split_location(float(split))
        new_outs = []
        new_ems = []
        new_imps = []

        # Build emulators for each cluster
        for c in range(cl.n_groups):
            cl_samples = default_selector(locations[clusters.labels_ == c], self._simulator_budget)
            cl_runs = self._sim_function(cl_samples)
            emulator = self._emulator_model()
            valid = self._diagnostic(emulator, cl_samples, cl_runs)
            if not valid:
                print(f"Emulation failed in wave {self._num_waves + 1}")
                # What do we do here?
            emulator.train(cl_samples, cl_runs)

            # Because we are in 1D, we can isolate this region and resample locally
            subspace = self._space.sample_to_region_limits(clusters.cluster_centers_[c])
            v = Variable(f"emulated_x_w{self._num_waves + 1}_c{c}", subspace[0], subspace[1])
            next_samples = default_designer(v, self._emulator_budget)

            # Reject implausible samples
            next_outputs = emulator.evaluate(next_samples)
            impl = implausibility(self._target_mean, next_outputs[0], self._target_variance, next_outputs[1])

            new_outs.extend(next_samples)
            new_ems.extend(next_outputs)
            new_imps.extend(impl)
            self._emulators.append(emulator)

        self._out_samples = np.asarray(new_outs)
        self._out_emulation = np.asarray(new_ems)
        self._out_imp = np.asarray(new_imps)

    def plot_current(self):
        if self._out_samples is None:
            print("Nothing to plot")
            return
        plot_1d_nroy(self._variable.min_support, self._variable.max_support, self._out_samples, self._out_imp, 100, 3.0)

    def plot_emulators(self):

        graph = HGraph()
        graph.set_dimensions(2, 1)

        x_, y_, i_ = (np.asarray(t).reshape(-1, 1)
                      for t in zip(*sorted(zip(self._out_samples, self._out_emulation, self._out_imp))))

        graph.axes[0] = plot_emulator_for_wave(graph.axes[0], 1, self._target_mean, self._target_variance,
                                               self._variable.min_support, self._variable.max_support,
                                               [x_], [y_[0]], [y_[1]])
        graph.axes[0] = plot_emulator_design_points(graph.axes[0], self.sim_design_points, self.sim_runs)

        # Implausibility trace
        graph.axes[1].scatter(x_, i_, 'b+')
        graph.axes[1].set(xlabel='x', ylabel='I(x)', title='Implausibility')
        graph.axes[1].hlines(y=3.0, xmin=self._variable.min_support, xmax=self._variable.max_support,
                             linewidth=1, color='k', linestyle='dashed')
        graph.axes[0].grid(linestyle='--', color='grey', alpha=0.5)
        graph.axes[1].grid(linestyle='--', color='grey', alpha=0.5)
        graph.plot()
