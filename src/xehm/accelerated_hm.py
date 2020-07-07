#
# Accelerated History Matching for low dimensional problems
#

#
# One dimensional cases
#
# In 1D history matching is likely extremely inefficient compared to other methods, but is valid
# Regions and space splits in 1D are analytically trivial, so cascading samples through the entire
# emulator hierarchy can be avoided for a simpler and faster implementation.
#
# 1D Cases should use some kind of root finding algorithm - False position is probably the best
#

from .space_descriptor import RegionOneDimensional
from .clustering import XMeans
from .emulators import Emulator
from .emulators import GaussianProcess
from .designs import default_designer, default_selector
from .diagnostics import LeaveOneOut
from .graphics import plot_1d_nroy, plot_emulator_for_wave, plot_emulator_design_points
from .graphics import HGraph
from ._sample import SISOSampleSet
from ._variables import Variable
from .utils import implausibility, print_separator_line, print_header
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

        # Problem description
        self._variable = Variable("x", min_x, max_x)
        self._target_mean = z_mean
        self._target_variance = z_variance
        self._space = RegionOneDimensional(self._variable.min_support, self._variable.max_support)
        self._num_waves = -1

        # Wave-space description
        self._emulators: List[Emulator] = []
        self._levels: List[int] = []
        self._clusters = []

        self._splits: List[List[float]] = []
        self._current_samples: Union[SISOSampleSet, None] = None

        # Sampling controls
        self._emulator_budget: int = 100
        self._simulator_budget: int = 10
        self.sim_design_points = []
        self.sim_runs = []

        # Analysis components
        self._sim_function = None
        self._emulator_model = emulator_model
        self._designer = default_designer
        self._diagnostic = LeaveOneOut.cross_validate

        # Outputs / data for plotting
        self._samples = None

    def set_simulator(self, simulator):
        self._sim_function = simulator

    def set_budgets(self, sim_max: int, em_max: int):
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

        print_header("Initialising a new history matching process")

        # If n_points is specified then update the simulator budget
        if n_points is not None:
            print(f"Setting simulator budget to {n_points}")
            self._simulator_budget = n_points
        else:
            print(f"Using previous simulator budget of {self._simulator_budget}")

        # If there is no simulator defined then we cannot run
        if self._sim_function is None:
            raise ValueError("No simulator defined for this history matching process")

        # Create the initial design points using the simulator
        # The budget is passed to the designer to limit the number of simulator evaluations
        initial_design = self._designer(self._variable, self._simulator_budget)
        initial_runs = self._sim_function(initial_design)
        print(f"Constructed a design of {initial_design.shape[0]} points")

        # Build and validate the emulator
        emulator = self._emulator_model()
        print(f"Constructed an emulator of type {emulator.ident}")
        valid = self._diagnostic(emulator, initial_design, initial_runs)
        if not valid:
            print(f"Emulator failed diagnostics in initial wave")
            # What do we do here?
        emulator.train(initial_design, initial_runs)
        print("Emulator has passed diagnostics")
        emulator.ident = "wave0_em0"

        # Generate emulation samples
        emulator_x = self._designer(self._variable, self._emulator_budget)
        means, variances = emulator.evaluate(emulator_x)
        emulator_i = implausibility(self._target_mean, means, self._target_variance, variances)
        print(f"Constructed implausibility set containing {self._emulator_budget} points")

        # Don't select out final samples to allow full space plotting
        self._samples = np.zeros((self._emulator_budget, 5))
        self._samples[:, 0] = emulator_x[:, 0]
        self._samples[:, 1] = means[:, 0]
        self._samples[:, 2] = variances[:, 0]
        self._samples[:, 3] = emulator_i[:, 0]
        self._samples[:, 4] = np.zeros(self._emulator_budget)
        self._emulators = [emulator]
        self._levels = [0]
        self._splits = []
        self._clusters = [None]
        self._num_waves = 0

    def run_wave(self, i_cut_off=3.0):

        # Check for calling before initialisation
        if self._num_waves < 0:
            print("WARNING: run_wave() called with no initialisation")
            print("Using defaults to initialise the run")
            self.initialise()

        self._num_waves += 1
        print_header(f"Running history matching wave {self._num_waves}")

        # Check the implausibility space and filter out 'good' samples
        non_imp = self._samples[self._samples[:, 3] <= i_cut_off]
        print(f"Previous wave contains {self._samples.shape[0]} samples, {non_imp.shape[0]} are non-implausible")

        # Numpy ruins 1D slices, so weld the dimension back on
        locations = non_imp[:, 0].reshape(-1, 1)

        # Each wave starts by inspecting the current available samples and discovering structure
        # np.newaxis stops numpy chopping off the dimensions
        cl = XMeans().assign(locations, None)
        print(f"Assigning points to space: recommending {cl.n_groups} clusters")

        # TODO: Do we need to call KMeans again??
        clusters = KMeans(n_clusters=cl.n_groups).fit(locations)
        for i, c in enumerate(clusters.cluster_centers_):
            print(f"Cluster {i + 1} centroid: {c[0]}")
        self._clusters.append(clusters)

        # Split space using cluster centres
        splits = []
        self._splits.append([])
        locs = sorted(clusters.cluster_centers_)
        if cl.n_groups > 1:
            splits = [0.5 * (locs[i + 1] + locs[i]) for i in range(cl.n_groups - 1)]
            self._splits[-1] = splits
        for split in splits:
            print(f"Split space at {split}")
            self._space.split_location(float(split))

        # Build emulators for each cluster
        for c in range(cl.n_groups):
            print_separator_line()
            print(f"Constructing emulator for cluster {c + 1}")
            samples_in_region = locations[clusters.labels_ == c]
            print(f"Found {samples_in_region.shape[0]} out of {locations.shape[0]} in this region")
            cl_samples = default_selector(locations[clusters.labels_ == c], self._simulator_budget)
            print(f"Selected {cl_samples.shape[0]} to build training sample")
            cl_runs = self._sim_function(cl_samples)

            emulator = self._emulator_model()
            print(f"Constructed an emulator of type {emulator.ident}")
            valid = self._diagnostic(emulator, cl_samples, cl_runs)
            if not valid:
                print(f"Emulator failed diagnostics")
                # What do we do here?
            emulator.train(cl_samples, cl_runs)
            print("Emulator has passed diagnostics")
            emulator.ident = f"wave{self._num_waves}_em{c}"

            # Because we are in 1D, we can isolate this region and resample locally
            #subspace = self._space.sample_to_region_limits(clusters.cluster_centers_[c])
            #v = Variable(f"emulated_x_w{self._num_waves + 1}_c{c}", subspace[0], subspace[1])
            #next_samples = default_designer(v, self._emulator_budget)
            #print(f"Sampled {self._emulator_budget} samples from [{subspace[0]}, {subspace[1]}]")

            # Store the new emulator and depth
            self._emulators.append(emulator)
            self._levels.append(self._num_waves)

        ## PREVIOUS INDENT
        # Generate new samples for the next iteration
        x, m, v, i, c = self.cascade_rejection_sampler(self._emulator_budget)

        self._samples = np.zeros((x.shape[0], 5))
        self._samples[:, 0] = x.squeeze()
        self._samples[:, 1] = m.squeeze()
        self._samples[:, 2] = v.squeeze()
        self._samples[:, 3] = i.squeeze()
        self._samples[:, 4] = c.squeeze()


    # Cascade rejection sampling is the standard method for history matching
    # We generate samples at the highest level and test against each emulation wave - samples are rejected
    # if they do not belong to the non-implausible set
    def cascade_rejection_sampler(self, budget: int):

        if not self._emulators:
            raise ValueError("History matching not initialised")

        # Resample from initial setup
        start_x = self._designer(self._variable, budget)
        start_m, start_v = self._emulators[0].evaluate(start_x)
        start_i = implausibility(self._target_mean, start_m, self._target_variance, start_v)

        # Filter from wave 0
        x = start_x[start_i <= 3.0].reshape(-1, 1)
        if x.shape[0] == 0:
            return np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))

        base = 1
        for wave in range(1, self._num_waves + 1):

            x_new = []
            m_new = []
            v_new = []
            i_new = []
            c_new = []

            # Calculate the cluster labels for each sample at this wave level
            ids = self._clusters[wave].predict(x)
            num_groups = len(self._clusters[wave].cluster_centers_)
            for g in range(num_groups):
                filtered_by_cluster = x[ids == g]
                emulator_index: int = g + base
                model = self._emulators[emulator_index]
                print(f"Cascading through: {model.ident}")
                m, v = model.evaluate(filtered_by_cluster)
                i = implausibility(self._target_mean, m, self._target_variance, v)
                mask = i <= 3.0
                accepted = filtered_by_cluster[i <= 3.0]
                x_new.extend(accepted)
                m_new.extend(m[mask])
                v_new.extend(v[mask])
                i_new.extend(i[mask])
                c_new.extend([emulator_index] * accepted.shape[0])
            base += num_groups
            x = np.asarray(x_new).reshape(-1, 1)

        m = np.asarray(m_new).reshape(-1, 1)
        v = np.asarray(v_new).reshape(-1, 1)
        i = np.asarray(i_new).reshape(-1, 1)
        c = np.asarray(c_new).reshape(-1, 1)
        return x, m, v, i, c


    def plot_current(self, resolution: Union[int, None]=None, i_cut_off=3.0):
        if self._samples is None:
            print("Nothing to plot")
            return

        s_count: int = int(self._emulator_budget / 10.0)
        if resolution is not None:
            s_count = resolution

        locs = self._samples[:, 0]
        imps = self._samples[:, 3]
        plot_1d_nroy(self._variable.min_support, self._variable.max_support, locs, imps, s_count, i_cut_off)

    def plot_emulators(self):

        if self._num_waves < 0:
            raise ValueError("No emulators to plot!")

        graph = HGraph()
        graph.set_dimensions(2, 1)

        # Samples array is: x, m, v, i as columns, sort on x as they are random draws
        plot_source = self._samples[self._samples[:, 0].argsort()]

        num_emulators = 1 if self._num_waves == 0 else len(self._splits[-1]) + 1
        for n in range(num_emulators):

            index = len(self._emulators) - 1 - n

            emulator_design_inputs = self._emulators[index]._design_inputs
            emulator_design_outputs = self._emulators[index]._design_outputs

            mask = plot_source[:, 4] == index
            x_ = plot_source[mask, 0]
            m_ = plot_source[mask, 1]
            v_ = plot_source[mask, 2]
            i_ = plot_source[mask, 3]

            graph.axes[0] = plot_emulator_for_wave(graph.axes[0], self._num_waves, self._target_mean, self._target_variance,
                                                   self._variable.min_support, self._variable.max_support,
                                                   [x_], [m_], [v_])

            # Add the training data to the plot
            graph.axes[0] = plot_emulator_design_points(graph.axes[0], emulator_design_inputs, emulator_design_outputs)

            # Implausibility trace
            graph.axes[1].scatter(x_, i_, c="b", marker='+')
            graph.axes[1].set(xlabel='x', ylabel='I(x)', title='Implausibility')
            graph.axes[1].hlines(y=3.0, xmin=self._variable.min_support, xmax=self._variable.max_support,
                                 linewidth=1, color='k', linestyle='dashed')
            graph.axes[0].grid(linestyle='--', color='grey', alpha=0.5)
            graph.axes[1].grid(linestyle='--', color='grey', alpha=0.5)
        graph.plot()
