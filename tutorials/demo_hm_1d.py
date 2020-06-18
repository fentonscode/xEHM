import xehm as hm
import numpy as np


# Simulator: This is the forward model which is to undergo calibration / analysis
def simulator(x) -> np.ndarray:
    return np.add(x, np.cos(np.multiply(10.0, x)))


def main():

    min_x = 0.0
    max_x = 1.0
    z_mean = 0.5
    z_variance = 0.01

    hmatch = hm.HistoryMatching1D(min_x, max_x, z_mean, z_variance)
    hmatch.set_simulator(simulator)
    hmatch.initialise(5)
    hmatch.plot_current()
    hmatch.plot_emulators()

    hmatch.run_wave()
    hmatch.plot_current()
    hmatch.run_wave()

    pass


if __name__ == '__main__':
    main()