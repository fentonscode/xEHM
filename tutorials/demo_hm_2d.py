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
    hmatch.set_budgets(10, 100000)

    hmatch.initialise()
    hmatch.plot_samples()

    # Do 5 waves
    for w in range(5):

        hmatch.run_wave()
        hmatch.plot_samples()
        #hmatch.plot_current(resolution=100)
        #hmatch.plot_emulators()

    print("Finished history matching")

    # Print final samples
    for s in hmatch._samples:
        print(f"x: {s[0]} | i: {s[3]} | c: {s[4]}")


if __name__ == '__main__':
    main()