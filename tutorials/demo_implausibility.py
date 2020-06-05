#
# This demo shows how to use the library to calculate an implausibility field for a simple
# 1 input, 1 output model
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import xehm as hm


# Simulator: This is the forward model which is to undergo calibration / analysis
def simulator(x) -> np.ndarray:
    return np.add(x, np.cos(np.multiply(10.0, x)))


# Implausibility: This is the metric by which potential inputs are ranked
def implausibility(match: float, expectation: np.ndarray, variance: float,
                   emulator_variance: np.ndarray) -> np.ndarray:
    return np.divide(np.abs(np.subtract(match, expectation)), np.sqrt(np.add(variance, emulator_variance)))


# Main demo starts here
def demo_implausibility():

    # We have observed z as 0.5 + N(0, 0.05^0.5)
    z_match = 0.5
    z_variance = 0.05

    # We assume that the single input is between 0 and 1
    x_min = 0.0
    x_max = 1.0

    # Pick a random design to start from
    initial_design = np.random.uniform(low=x_min, high=x_max, size=10).reshape(-1, 1)
    initial_runs = simulator(initial_design).reshape(-1, 1)

    # Generate an emulator
    emulator = hm.emulators.GaussianProcess().build(initial_design, initial_runs)

    # Linearly space a lot of design points and calculate implausibility
    x_test = np.linspace(x_min, x_max, 1000)
    y_mean, y_variance = emulator.evaluate(x_test)

    # Calculate implausibility
    hm_cutoff = 3.0
    impl = implausibility(z_match, np.asarray(y_mean), z_variance, np.asarray(y_variance))

    # Plot the problem graphically
    z_min = z_match - (2.0 * np.sqrt(z_variance))
    z_max = z_match + (2.0 * np.sqrt(z_variance))

    fig = plt.figure(figsize=(7, 6))
    gs = grid.GridSpec(2, 1)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax1.plot(x_test, y_mean, color='b')
    ax1.fill_between(x_test, y_mean[:, 0] - 2.0 * np.sqrt(y_variance[:, 0]),
                     y_mean[:, 0] + 2.0 * np.sqrt(y_variance[:, 0]), color="C0", alpha=0.2)
    ax1.plot(initial_design, initial_runs, 'kx')
    ax1.set(xlabel='x', ylabel='z', title='Emulator output')
    ax1.fill_between(x=x_test.ravel(), y1=z_min, y2=z_max, color='green', alpha=0.25)
    ax1.hlines(y=z_match, xmin=x_min, xmax=x_max, color='k', linestyle='dashed')
    ax2.plot(x_test, impl)
    ax2.set(xlabel='x', ylabel='I(x)', title='Implausibility')
    ax2.hlines(y=hm_cutoff, xmin=x_min, xmax=x_max, linewidth=1, color='k', linestyle='dashed')
    ax1.grid(linestyle='--', color='grey', alpha=0.5)
    ax2.grid(linestyle='--', color='grey', alpha=0.5)
    fig.show()


if __name__ == '__main__':
    demo_implausibility()

