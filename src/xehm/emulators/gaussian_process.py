import numpy as np
import gpflow as gp
from .emulator import Emulator

__all__ = ["GaussianProcess"]


class GaussianProcess(Emulator):
    def __init__(self, mean_function=None, kernel=gp.kernels.SquaredExponential()):
        super().__init__()
        self.ident = "Gaussian Process Emulator"
        self.kernel = kernel
        self.mean_function = mean_function
        self.model = None

    def train(self, inputs: np.ndarray, outputs: np.ndarray, parameter_defaults=None):

        if parameter_defaults is None:
            default_lh = 0.01
            default_ls = [0.1] * inputs.shape[1]
        else:
            default_lh = parameter_defaults[0]
            default_ls = parameter_defaults[1]

        self.kernel = gp.kernels.SquaredExponential(lengthscales=default_ls)

        # SE Kernel and regression model
        # NOTE: GPFlow documentation is notoriously poor - it might be worth checking the source code
        # NOTE: The LH model is a gaussian, but only a single variance parameter
        self.model = gp.models.GPR(mean_function=self.mean_function, kernel=self.kernel,
                                   data=(inputs, outputs))

        # Initial training parameters
        self.model.likelihood.variance.assign(default_lh)
        self.model.kernel.lengthscales.assign(default_ls)

        opt = gp.optimizers.Scipy()
        opt_logs = opt.minimize(self.model.training_loss, self.model.trainable_variables,
                                options=dict(maxiter=100))

        self._design_inputs = inputs
        self._design_outputs = outputs
        return self

    def evaluate(self, points):
        mean, var = self.model.predict_f(points)
        return np.asarray(mean), np.asarray(var)

    # Unused for now
    def validate(self, inputs: np.ndarray, outputs: np.ndarray, diag):
        print(f"Constructed an emulator of type {self.ident}")
        valid = diag(self, inputs, outputs)
        if not valid:
            print(f"Emulator failed diagnostics in initial wave")
            # What do we do here?
        self.train(inputs, outputs)
        print("Emulator has passed diagnostics")

