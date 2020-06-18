#
# This demo shows how to use the library to calculate an implausibility field for a simple
# 1 input, 1 output model
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import xehm as hm
from sklearn.cluster import KMeans
from typing import Union, List
from xehm.designs import default_designer
from xehm.utils import implausibility


class emulator_node:
    def __init__(self):
        self.model: Union[hm.emulators.Emulator, None] = None
        self.children: List[hm.emulators.Emulator] = []


# Simulator: This is the forward model which is to undergo calibration / analysis
def simulator(x) -> np.ndarray:
    return np.add(x, np.cos(np.multiply(10.0, x)))


# Main demo starts here
def demo_implausibility():

    # Complete a 3 wave history match

    # We have observed z as 0.5 + N(0, 0.01^0.5)
    z_match = 0.5
    z_variance = 0.01

    # Variables to track across the waves
    head_node: emulator_node = emulator_node()

    # We assume that the single input is between 0 and 1
    input_var = hm.Variable("x", 0.0, 1.0)

    # Wave 1
    # ------

    # Pick a random design to start from
    initial_design = default_designer(input_var, 5)
    initial_runs = simulator(initial_design)

    # Generate an emulator for wave 1
    head_node.model = hm.emulators.GaussianProcess()
    cv = hm.diagnostics._leave_one_out.cross_validate(head_node.model, initial_design, initial_runs)
    #if not cv:
    #    print("Wave 1 emulator failed validation")
    #    # What do we do here?
    #    # Try re-drawing the input samples
    #    initial_design = np.random.uniform(low=x_min, high=input_var.max_support, size=5).reshape(-1, 1)
    #    initial_runs = simulator(initial_design).reshape(-1, 1)
    #    head_node.model = hm.emulators.GaussianProcess()
    #    cv = hm.diagnostics._leave_one_out.cross_validate(head_node.model, initial_design, initial_runs)
    #    if not cv:
    #        raise Exception("Bad emulators")

    head_node.model.train(initial_design, initial_runs)

    # Sample the support uniformly and calculate implausibility
    x_test = np.random.uniform(input_var.min_support, input_var.max_support, 1000).squeeze()
    x_test.sort()
    y_mean, y_variance = head_node.model.evaluate(x_test)

    # Calculate implausibility and filter ones that are non-implausible
    hm_cutoff = [3.0, 1.0] # Aggressive roll-off in wave 2
    impl = implausibility(z_match, np.asarray(y_mean), z_variance, np.asarray(y_variance))

    hm.graphics.plot_1d_nroy(input_var.min_support, input_var.max_support, x_test.reshape(-1, 1), impl, 100, 3.0)

    non_imp = np.asarray([(x_test[i], i) for i, imp in enumerate(impl) if imp <= hm_cutoff[0]])
    print(f"Wave 1 emulator produced {len(non_imp)} non-implausible samples")
    index = [int(n[1]) for n in non_imp]
    non_imp = np.asarray([n[0] for n in non_imp])

    cl = hm.clustering.XMeans().assign(non_imp.reshape(-1, 1), None)
    print(f"Recommending {len(cl)} clusters")

    # TODO: Move to external test
    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(111)
    #ax.plot(range(1, k + 1), b_trace, 'k-o')
    #fig.show()

    clusters = KMeans(n_clusters=3).fit(non_imp.reshape(-1, 1))

    # Plot the wave
    z_min = z_match - (2.0 * np.sqrt(z_variance))
    z_max = z_match + (2.0 * np.sqrt(z_variance))

    fig = plt.figure(figsize=(7, 6))
    gs = grid.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    ax1 = hm.graphics.plot_emulator_for_wave(ax1, 1, z_match, z_variance, input_var.min_support, input_var.max_support, [x_test], [y_mean], [y_variance])
    ax1 = hm.graphics.plot_emulator_design_points(ax1, initial_design, initial_runs)

    ax2.plot(x_test, impl)
    ax2.plot(non_imp, impl[index], 'gx')
    ax2.set(xlabel='x', ylabel='I(x)', title='Implausibility')
    ax2.hlines(y=hm_cutoff[0], xmin=input_var.min_support, xmax=input_var.max_support, linewidth=1, color='k', linestyle='dashed')
    ax1.grid(linestyle='--', color='grey', alpha=0.5)
    ax2.grid(linestyle='--', color='grey', alpha=0.5)
    fig.show()

    # Wave 2
    # ------

    # There are 3 clusters in wave 2
    w2_variances = []
    hm_samples = []
    hm_impl = []

    fig = plt.figure(figsize=(7, 6))
    gs = grid.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    for i in range(3):

        # Use the samples that made it through wave 1 to train wave 2
        emulator_samples = non_imp[clusters.labels_ == i].reshape(-1, 1)
        r_new = simulator(emulator_samples).reshape(-1, 1)
        w2_emulator = hm.emulators.GaussianProcess().train(emulator_samples, r_new)
        head_node.children.append(w2_emulator)

        # Sample again from subset to get average variance for graphs
        y2_new, y2_var = w2_emulator.evaluate(emulator_samples)
        w2_variances.append(y2_var)

        # Sample again from top
        x2_test = np.random.uniform(input_var.min_support, input_var.max_support, 333).squeeze()
        x2_test.sort()
        y1_mean, y1_variance = head_node.model.evaluate(x2_test)
        impl = implausibility(z_match, np.asarray(y1_mean), z_variance, np.asarray(y1_variance))
        non_imp_2w1 = np.asarray([x2_test[i] for i, imp in enumerate(impl) if imp <= hm_cutoff[0]])

        # Cascade down to wave two
        cluster_id = clusters.predict(non_imp_2w1.reshape(-1, 1))
        non_imp_w2 = non_imp_2w1[cluster_id == i]
        y2_mean, y2_variance = w2_emulator.evaluate(non_imp_w2)
        impl = implausibility(z_match, np.asarray(y2_mean), z_variance, np.asarray(y2_variance))
        non_imp_2w2 = np.asarray([(non_imp_w2[i], i) for i, imp in enumerate(impl) if imp <= hm_cutoff[1]])
        index = [int(n[1]) for n in non_imp_2w2]
        non_imp_2w2 = np.asarray([n[0] for n in non_imp_2w2])

        hm_samples.extend(non_imp_2w2)
        hm_impl.extend(impl[index])

        xplot = non_imp_w2#np.sort(non_imp)
        ax1.plot(xplot, y2_mean[:, 0], 'b')
        ax1.fill_between(xplot, y2_mean[:, 0] - 2.0 * np.sqrt(y2_variance[:, 0]),
                         y2_mean[:, 0] + 2.0 * np.sqrt(y2_variance[:, 0]), color="C0", alpha=0.2)
        ax1.plot(emulator_samples, r_new, 'kx')
        ax1.fill_between(x=xplot.ravel(), y1=z_min, y2=z_max, color='green', alpha=0.25)
        ax2.plot(xplot, impl, 'b')
        ax2.plot(non_imp_2w2, impl[index], 'gx')

    hm.graphics.plot_1d_nroy(input_var.min_support, input_var.max_support, np.asarray(hm_samples).reshape(-1, 1), np.asarray(hm_impl).reshape(-1, 1), 100, 1.0)

    ax1.set(xlabel='x', ylabel='z', title=f'Emulator output - wave 2')
    ax1.hlines(y=z_match, xmin=input_var.min_support, xmax=input_var.max_support, color='k', linestyle='dashed')
    ax2.set(xlabel='x', ylabel='I(x)', title='Implausibility')
    ax2.hlines(y=hm_cutoff[1], xmin=input_var.min_support, xmax=input_var.max_support, linewidth=1, color='k', linestyle='dashed')
    ax1.grid(linestyle='--', color='grey', alpha=0.5)
    ax2.grid(linestyle='--', color='grey', alpha=0.5)
    fig.show()

    print(f"Wave 2 emulator produced {len(hm_samples)} non-implausible samples")
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    ax1.hist(hm_samples, bins=len(hm_samples))
    fig.show()

    fig = plt.figure(figsize=(10, 10))
    gs = grid.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    ax1.hist(y_variance, bins=50)
    ax2.hist(w2_variances[0], bins=50)
    ax3.hist(w2_variances[1], bins=50)
    ax4.hist(w2_variances[2], bins=50)
    fig.show()


if __name__ == '__main__':
    demo_implausibility()
