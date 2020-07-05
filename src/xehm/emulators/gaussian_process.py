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

    def train(self, inputs: np.ndarray, outputs: np.ndarray, parameter_defaults=(0.01, 0.1)):
        # SE Kernel and regression model
        self.model = gp.models.GPR(mean_function=self.mean_function, kernel=self.kernel,
                                   data=(inputs, outputs))

        # Initial training parameters
        self.model.likelihood.variance.assign(parameter_defaults[0])
        self.model.kernel.lengthscales.assign(parameter_defaults[1])

        opt = gp.optimizers.Scipy()
        opt_logs = opt.minimize(self.model.training_loss, self.model.trainable_variables,
                                options=dict(maxiter=100))

        self._design_inputs = inputs
        self._design_outputs = outputs
        return self

    def evaluate(self, points):
        mean, var = self.model.predict_f(points.reshape(-1, 1))
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

