import xehm as hm
import numpy as np
from xehm._distributions import Tree2D


dist = Tree2D()


# Simulator: This is the forward model which is to undergo calibration / analysis
def simulator(x) -> (int, np.ndarray):
    return hm.utils.plugin.ReturnState.ok, dist.probability(x)


def main():

    min_x = 0.0
    max_x = 1.0
    z_mean = 5
    z_variance = 0.1

    hmatch = hm.HistoryMatching2D(min_x, max_x, z_mean, z_variance)
    hmatch.set_simulator(simulator)
    hmatch.set_budgets(50, 100000)

    hmatch.initialise()
    hmatch.plot_samples()

    # Do 3 waves
    for w in range(3):

        hmatch.run_wave()
        hmatch.plot_samples()

    print("Finished history matching")

    # Print final samples
    for s in zip(hmatch.x, hmatch.i, hmatch.c):
        print(f"x: {s[0]} | i: {s[1]} | c: {s[2]}")


if __name__ == '__main__':
    main()