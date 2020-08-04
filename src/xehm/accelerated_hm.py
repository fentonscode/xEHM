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
from .diagnostics import LeaveOneOut, LeaveOneOutStrict, leave_one_out
from .graphics import plot_1d_nroy, plot_emulator_for_wave, plot_emulator_design_points
from .graphics import HGraph, plot_2d_samples
from ._sample import SISOSampleSet
from ._variables import Variable, make_variable_set
from .utils import build_custom_plugin, implausibility, print_separator_line, print_header, ReturnState
from ._exceptions import SimulationFailure
from typing import Union, List
from sklearn.cluster import KMeans
import numpy as np

__all__ = ["HistoryMatching1D", "HistoryMatching2D"]


class HMBase:
    def __init__(self, emulator_budget: int = 1000, input_dimensions: int = 1, simulator_budget: int = 10,
                 emulator_model: Emulator = GaussianProcess, design_process=default_designer,
                 diagnostics=leave_one_out, simulator_function=None):

        # Simulator component - required
        self._sim_function = simulator_function
        if simulator_function is not None:
            self._sim_function = build_custom_plugin(simulator_function)

        # Input designer
        self._designer = build_custom_plugin(design_process)

        # Diagnostics
        self._diagnostic = build_custom_plugin(diagnostics)

        self._emulator_budget = emulator_budget
        self._simulator_budget = simulator_budget
        self._n_input_dims = input_dimensions
        self._samples = None
        self._num_waves = -1
        self._emulators: List[Emulator] = []

        # Analysis components
        self._emulator_model = emulator_model

        # Performance stats
        self._n_sim_calls = 0
        self._n_emulators_built = 0
        self._n_emulator_calls = 0
        self._n_rejected_samples = 0

        # Sample state
        self.x = None
        self.i = None

    def set_simulator(self, simulator):
        self._sim_function = build_custom_plugin(simulator)

    def set_budgets(self, sim_max: int, em_max: int):
        self._simulator_budget = sim_max
        self._emulator_budget = em_max

    def get_posterior_samples(self):
        return self._samples

    def call_simulator(self, x):
        # If there is no simulator defined then we cannot run
        if self._sim_function is None:
            raise ValueError("No simulator defined for this history matching process")

        result, data = self._sim_function(x=x)
        if result != ReturnState.ok:
            print(f"Simulator failed: {result}")
            raise SimulationFailure("Simulator unable to produce suitable outout")
        return data

    def print_performance(self):
        print_header("History matching performance summary")
        print(f"{self._n_sim_calls} simulator runs used")
        print(f"{self._n_emulators_built} emulators constructed")
        print(f"{self._n_emulator_calls} emulator runs used")
        print(f"{self._n_rejected_samples} rejections")
        print_separator_line()

        a: float = (self._n_emulator_calls - self._n_rejected_samples) / float(self._n_emulator_calls)
        print(f"Overall acceptance ratio: {a:.6f}")

    def plot_samples(self):

        if self.x is None:
            raise ValueError("No samples to plot!")

        if self.i is None:
            print("Warning: No implausibility data available, plot data will not be classified")

        n_dims = self.x.shape[1]
        if n_dims > 2:
            raise ValueError("Too many dimensions to plot")

        if n_dims == 1:
            pass
        if n_dims == 2:
            # NOTE: The run_wave function culls implausible automatically
            plot_2d_samples(self.x)
            pass


# Parameters
#   - Minimum support
#   - Maximum support
#   - Calibration mean
#   - Calibration variance
#   - Simulator
#   - Emulator (or use default)
#   - Designer (or use default)

class HistoryMatching1D(HMBase):
    def __init__(self, min_x, max_x, z_mean, z_variance):
        super().__init__(emulator_budget=100, simulator_budget=10)

        # Problem description
        self._variable = Variable("x", min_x, max_x)
        self._target_mean = z_mean
        self._target_variance = z_variance
        self._space = RegionOneDimensional(self._variable.min_support, self._variable.max_support)

        # Wave-space description
        self._levels: List[int] = []
        self._clusters = []

        self._splits: List[List[float]] = []
        self._current_samples: Union[SISOSampleSet, None] = None

        # Sampling controls
        self.sim_design_points = []
        self.sim_runs = []

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
        initialise_wave_zero(self, n_points)

    #        # Create the initial design points using the simulator
    #        # The budget is passed to the designer to limit the number of simulator evaluations
    #        initial_design = self._designer(variables=self._variable, points=self._simulator_budget)
    #        initial_runs = self.call_simulator(initial_design)
    #        print(f"Constructed a design of {initial_design.shape[0]} points")
    #
    #        # Build and validate the emulator
    #        emulator = self._emulator_model()
    #        print(f"Constructed an emulator of type {emulator.ident}")
    #        valid = self._diagnostic(emulator, initial_design, initial_runs)
    #        if not valid:
    #            print(f"Emulator failed diagnostics in initial wave")
    #            # What do we do here?
    #        emulator.train(initial_design, initial_runs)
    #        print("Emulator has passed diagnostics")
    #        emulator.ident = "wave0_em0"
    #        self._n_emulators_built += 1
    #
    #        # Generate emulation samples
    #        emulator_x = self._designer(self._variable, self._emulator_budget)
    #        means, variances = emulator.evaluate(emulator_x)
    #        self._n_emulator_calls += len(emulator_x)
    #        emulator_i = implausibility(self._target_mean, means, self._target_variance, variances)
    #        print(f"Constructed implausibility set containing {self._emulator_budget} points")
    #
    #        # Don't select out final samples to allow full space plotting
    #        self._samples = np.zeros((self._emulator_budget, 5))
    #        self._samples[:, 0] = emulator_x[:, 0]
    #        self._samples[:, 1] = means[:, 0]
    #        self._samples[:, 2] = variances[:, 0]
    #        self._samples[:, 3] = emulator_i[:, 0]
    #        self._samples[:, 4] = np.zeros(self._emulator_budget)
    #        self._emulators = [emulator]
    #        self._levels = [0]
    #        self._splits = []
    #        self._clusters = [None]
    #        self._num_waves = 0

    def run_wave(self, i_cut_off=3.0):

        # Check for calling before initialisation
        if self._num_waves < 0:
            print("WARNING: run_wave() called with no initialisation")
            print("Using defaults to initialise the run")
            initialise_wave_zero(self)

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
            cl_runs = self.call_simulator(cl_samples)

            emulator = self._emulator_model()
            print(f"Constructed an emulator of type {emulator.ident}")
            valid = self._diagnostic(emulator, cl_samples, cl_runs)
            if not valid:
                print(f"Emulator failed diagnostics")
                # What do we do here?
            emulator.train(cl_samples, cl_runs)
            print("Emulator has passed diagnostics")
            emulator.ident = f"wave{self._num_waves}_em{c}"
            self._n_emulators_built += 1

            # Because we are in 1D, we can isolate this region and resample locally
            # subspace = self._space.sample_to_region_limits(clusters.cluster_centers_[c])
            # v = Variable(f"emulated_x_w{self._num_waves + 1}_c{c}", subspace[0], subspace[1])
            # next_samples = default_designer(v, self._emulator_budget)
            # print(f"Sampled {self._emulator_budget} samples from [{subspace[0]}, {subspace[1]}]")

            # Store the new emulator and depth
            self._emulators.append(emulator)
            self._levels.append(self._num_waves)

        ## PREVIOUS INDENT
        # Generate new samples for the next iteration
        x, m, v, i, c = cascade_rejection_sampler(self, self._emulator_budget, 1)

        self._samples = np.zeros((x.shape[0], 5))
        self._samples[:, 0] = x.squeeze()
        self._samples[:, 1] = m.squeeze()
        self._samples[:, 2] = v.squeeze()
        self._samples[:, 3] = i.squeeze()
        self._samples[:, 4] = c.squeeze()

    def plot_current(self, resolution: Union[int, None] = None, i_cut_off=3.0):
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

            graph.axes[0] = plot_emulator_for_wave(graph.axes[0], self._num_waves, self._target_mean,
                                                   self._target_variance,
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


#
# 2D Case
#
class HistoryMatching2D(HMBase):
    def __init__(self, min_x, max_x, z_mean, z_variance):
        super().__init__(emulator_budget=100, simulator_budget=10)

        # Problem description
        self._variable = make_variable_set(2, "x", min_x, max_x)
        self._target_mean = z_mean
        self._target_variance = z_variance
        self._num_waves = -1

        # Wave-space description
        self._levels: List[int] = []
        self._clusters = []

        self._splits: List[List[float]] = []

        # Sampling controls
        self.sim_design_points = []
        self.sim_runs = []

        # Generic N-way HM variables
        self.m = None
        self.v = None
        self.c = None

        self.diagnostic_settings = {"plot_report": True}

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
        initialise_wave_zero(self, n_points)

    def run_wave(self, i_cut_off: float = 3.0, allow_initialisation: bool = True):

        # Check for calling before initialisation
        if self._num_waves < 0:
            if allow_initialisation:
                print("WARNING: run_wave() called with no initialisation")
                print("Using defaults to initialise the run")
                initialise_wave_zero(self)
            else:
                raise ValueError("Cannot run wave, history matching not initialised")

        self._num_waves += 1
        print_header(f"Running history matching wave {self._num_waves}")

        # Check the implausibility space and filter out 'good' samples
        # The reshape command prevents rouge 1D slices from chopping dimensions off
        mask = self.i[:, 0] <= i_cut_off
        locations = self.x[mask].reshape(-1, self.x.shape[1])
        if locations.shape[0] == 0:
            # Q: What do we do if there are no samples?
            raise ValueError("No samples were non-implausible")

        print(f"Previous wave contains {self.x.shape[0]} samples, {locations.shape[0]} are non-implausible")

        # Each wave starts by inspecting the current available samples and discovering structure
        # np.newaxis stops numpy chopping off the dimensions
        cl = XMeans().assign(locations, None)
        print(f"Assigning points to space: recommending {cl.n_groups} clusters")

        # TODO: Do we need to call KMeans again??
        # FIXME: Remove this hack - it is just to speed up local testing
        recs = cl.n_groups
        if recs > 4:
            recs = 4
        clusters = KMeans(n_clusters=recs).fit(locations)
        for i, c in enumerate(clusters.cluster_centers_):
            print(f"Cluster {i + 1} centroid: {','.join([str(k) for k in c])}")
        self._clusters.append(clusters)

        from hdbscan import HDBSCAN as hd
        h = hd().fit(X=locations)
        print(f"HDBSCAN: {max(h.labels_)}")

        # Build emulators for each cluster
        for c in range(cl.n_groups):
            print_separator_line()
            print(f"Constructing emulator for cluster {c + 1}")
            samples_in_region = locations[clusters.labels_ == c]
            print(f"Found {samples_in_region.shape[0]} out of {locations.shape[0]} in this region")
            cl_samples = default_selector(locations[clusters.labels_ == c], self._simulator_budget)
            print(f"Selected {cl_samples.shape[0]} to build training sample")
            cl_runs = self.call_simulator(cl_samples)

            emulator = self._emulator_model()
            print(f"Constructed an emulator of type {emulator.ident}")
            valid = self._diagnostic(emulator_model=emulator, reference_inputs=cl_samples, reference_outputs=cl_runs,
                                     **self.diagnostic_settings)
            if not valid:
                print(f"Emulator failed diagnostics")
                # What do we do here?
            emulator.train(cl_samples, cl_runs)
            print("Emulator has passed diagnostics")
            emulator.ident = f"wave{self._num_waves}_em{c}"

            # Store the new emulator and depth
            self._emulators.append(emulator)
            self._levels.append(self._num_waves)

        # Generate new samples for the next iteration
        self.x, self.m, self.v, self.i, self.c = cascade_rejection_sampler(self, self._emulator_budget, 2)

    def plot_current(self, resolution: Union[int, None] = None, i_cut_off=3.0):
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

            graph.axes[0] = plot_emulator_for_wave(graph.axes[0], self._num_waves, self._target_mean,
                                                   self._target_variance,
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


#
# Initialise: Setup the first components for successive history matching waves
#
# n_points: This sets the simulator budget for the process.
# i_cut_off: implausibility cut-off for the initial samples
#
# Notes:
#   - If the simulator has not been defined / attached then this will fail
#
def initialise_wave_zero(match_job, n_points: Union[int, None] = None):
    print_header("Initialising a new history matching process")

    # If n_points is specified then update the simulator budget
    if n_points is not None:
        print(f"Setting simulator budget to {n_points}")
        match_job._simulator_budget = n_points
    else:
        print(f"Using previous simulator budget of {match_job._simulator_budget}")

    # If there is no simulator defined then we cannot run
    if match_job._sim_function is None:
        raise ValueError("No simulator defined for this history matching process")

    # Create the initial design points using the simulator
    # The budget is passed to the designer to limit the number of simulator evaluations
    initial_design = match_job._designer(variables=match_job._variable, points=match_job._simulator_budget)
    initial_runs = match_job.call_simulator(initial_design)
    print(f"Constructed a design of {initial_design.shape[0]} points")

    # Build and validate the emulator
    emulator = match_job._emulator_model()
    print(f"Constructed an emulator of type {emulator.ident}")
    valid = match_job._diagnostic(emulator_model=emulator, reference_inputs=initial_design,
                                  reference_outputs=initial_runs, debug_print=True)
    if not valid:
        print(f"Emulator failed diagnostics in initial wave")
        # What do we do here?
    emulator.train(initial_design, initial_runs)
    print("Emulator has passed diagnostics")
    emulator.ident = "wave0_em0"
    match_job._n_emulators_built += 1

    # Generate emulation samples
    emulator_x = match_job._designer(variables=match_job._variable, points=match_job._emulator_budget)
    means, variances = emulator.evaluate(emulator_x)
    match_job._n_emulator_calls += len(emulator_x)
    emulator_i = implausibility(match_job._target_mean, means, match_job._target_variance, variances)
    print(f"Constructed implausibility set containing {match_job._emulator_budget} points")

    # Save structure
    match_job._emulators = [emulator]
    match_job._levels = [0]
    match_job._splits = []
    match_job._clusters = [None]
    match_job._num_waves = 0

    # Save state
    match_job.x = emulator_x
    match_job.m = means
    match_job.v = variances
    match_job.i = emulator_i
    match_job.c = np.zeros(match_job._emulator_budget)


# Cascade rejection sampling is the standard method for history matching
# We generate samples at the highest level and test against each emulation wave - samples are rejected
# if they do not belong to the non-implausible set
def cascade_rejection_sampler(match_job, budget: int, dimensions: int = 1) -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    if not match_job._emulators:
        raise ValueError("History matching not initialised")

    # Resample from initial setup
    start_x = match_job._designer(variables=match_job._variable, points=budget)
    start_m, start_v = match_job._emulators[0].evaluate(start_x)
    match_job._n_emulator_calls += start_x.shape[0]
    start_i = implausibility(match_job._target_mean, start_m, match_job._target_variance, start_v)

    # Filter from wave 0
    x = start_x[start_i[:, 0] <= 3.0].reshape(-1, dimensions)
    if x.shape[0] == 0:
        return np.empty((0, dimensions)) * 5

    base = 1
    for wave in range(1, match_job._num_waves + 1):

        x_new = []
        m_new = []
        v_new = []
        i_new = []
        c_new = []

        # Calculate the cluster labels for each sample at this wave level
        ids = match_job._clusters[wave].predict(x)
        num_groups = len(match_job._clusters[wave].cluster_centers_)
        for g in range(num_groups):
            filtered_by_cluster = x[ids == g]
            emulator_index: int = g + base
            model = match_job._emulators[emulator_index]
            print(f"Cascading through: {model.ident}")
            m, v = model.evaluate(filtered_by_cluster)
            match_job._n_emulator_calls += filtered_by_cluster.shape[0]
            i = implausibility(match_job._target_mean, m, match_job._target_variance, v)
            mask = i[:, 0] <= 3.0
            accepted = filtered_by_cluster[mask]
            x_new.extend(accepted)
            m_new.extend(m[mask])
            v_new.extend(v[mask])
            i_new.extend(i[mask])
            c_new.extend([emulator_index] * accepted.shape[0])
        base += num_groups
        x = np.asarray(x_new).reshape(-1, dimensions)

    m = np.asarray(m_new).reshape(-1, 1)
    v = np.asarray(v_new).reshape(-1, 1)
    i = np.asarray(i_new).reshape(-1, 1)
    c = np.asarray(c_new).reshape(-1, 1)
    match_job._n_rejected_samples += (start_x.shape[0] - x.shape[0])

    return x, m, v, i, c
