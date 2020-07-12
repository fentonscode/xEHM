import xehm as hm
import numpy as np
from xehm._distributions import Tree2D


dist = Tree2D()

# Simulator: This is the forward model which is to undergo calibration / analysis
def simulator(x) -> np.ndarray:
    return dist.probability(x)


def main():

    min_x = 0.0
    max_x = 1.0
    z_mean = 5
    z_variance = 0.1

    hmatch = hm.HistoryMatching2D(min_x, max_x, z_mean, z_variance)
    hmatch.set_simulator(simulator)
    hmatch.set_budgets(50, 100000)

    # Load in a custom diagnostic suite, then call it to generate the functions for each test
    diagnostic_suite = hm.utils.build_custom_plugin("..\\customising\\plugin_diagnostic::diagnostic_none")
    diagnostic_functions = diagnostic_suite()
    hmatch._diagnostic = diagnostic_functions[0]

    hmatch.initialise()
    hmatch.plot_samples()

    # Do 5 waves
    for w in range(10):

        hmatch.run_wave()
        hmatch.plot_samples()

    print("Finished history matching")

    # Print final samples
    for s in zip(hmatch.x, hmatch.i, hmatch.c):
        print(f"x: {s[0]} | i: {s[1]} | c: {s[2]}")


if __name__ == '__main__':
    main()