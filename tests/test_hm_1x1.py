#
# Test/demo a one input one output history matching condition: WIP
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid


def emulator(x):
    return np.add(x, np.cos(np.multiply(10.0, x)))


def implausibility(match, expectation, variance):
    return np.divide(np.abs(np.subtract(match, expectation)), np.sqrt(variance))


def hm1x1():

    # Problem description
    z_match = 0.5
    z_variance = 0.05
    x_min = 0.0
    x_max = 1.0
    x_test = np.linspace(x_min, x_max, 1000)
    y_test = emulator(x_test)
    hm_cutoff = 3.0

    impl = implausibility(z_match, y_test, z_variance)

    # Plot the problem graphically
    fig = plt.figure(figsize=(7, 6))
    gs = grid.GridSpec(2, 1)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax1.plot(x_test, y_test)
    ax1.set(xlabel='x', ylabel='z', title='Emulator output')
    ax1.hlines(y=z_match, xmin=x_min, xmax=x_max, linewidth=1, color='r')
    ax2.plot(x_test, impl)
    ax2.set(xlabel='x', ylabel='I(x)', title='Implausibility')
    ax2.hlines(y=hm_cutoff, xmin=x_min, xmax=x_max, linewidth=1, color='r')
    ax1.grid(linestyle='--', color='grey', alpha=0.5)
    ax2.grid(linestyle='--', color='grey', alpha=0.5)
    fig.show()


if __name__ == '__main__':
    hm1x1()

