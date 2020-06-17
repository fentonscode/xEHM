import numpy as np

# Shut up TensorFlow and prevent it dominating the output
import tensorflow as tf

tf.get_logger().setLevel('INFO')

# gpflow will load tensorflow with too much console activity for most users
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

    def train(self, inputs, outputs, parameter_defaults=(0.01, 0.1)):
        # SE Kernel and regression model
        self.model = gp.models.GPR(mean_function=self.mean_function, kernel=self.kernel,
                                   data=(inputs, outputs))

        # Initial training parameters
        self.model.likelihood.variance.assign(parameter_defaults[0])
        self.model.kernel.lengthscales.assign(parameter_defaults[1])

        opt = gp.optimizers.Scipy()
        opt_logs = opt.minimize(self.model.training_loss, self.model.trainable_variables,
                                options=dict(maxiter=100))
        return self

    def evaluate(self, points):
        mean, var = self.model.predict_f(points.reshape(-1, 1))
        return np.asarray(mean), np.asarray(var)
