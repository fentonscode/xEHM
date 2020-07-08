# Leave-one-out (LOO) cross validation
# NOTE: scikit-learn has something called this, but it doesn't do what we need it to do

from ..emulators.emulator import Emulator
import numpy as np
from math import erf

__all__ = ["LeaveOneOut", "LeaveOneOutStrict"]


#
# Leave-one-out cross validation tests an emulators *structure* to see that it can generalise
#   - This will re-train the emulator, so to be safe we copy it
#

class LeaveOneOut:

    @staticmethod
    def cross_validate(emulator: Emulator, inputs: np.ndarray, outputs: np.ndarray) -> bool:
        n_samples = inputs.shape[0]
        errors = np.zeros((n_samples, outputs.shape[1]))
        variances = np.zeros((n_samples, outputs.shape[1]))
        model_outs = np.zeros((n_samples, outputs.shape[1]))
        for i in range(n_samples):
            mask = (np.arange(n_samples) != i)
            train_inputs = inputs[mask]
            train_outputs = outputs[mask]
            emulator.train(train_inputs, train_outputs)
            em_in = inputs[i].reshape(1, inputs.shape[1])
            model_outs[i], variances[i] = emulator.evaluate(em_in)
            errors[i] = model_outs[i] - outputs[i]
        delta = np.multiply(2.0, np.sqrt(variances))
        upper = np.add(outputs, delta)
        lower = np.subtract(outputs, delta)
        return not (np.any(model_outs > upper) or np.any(model_outs < lower))


class LeaveOneOutStrict:

    def __init__(self, emulator_model: Emulator = None, width: int = 2):
        self.model = emulator_model
        self.passed = False
        self.sigmas = width
        self._critical_failure_rate = 1.0 - erf(self.sigmas / np.sqrt(2.0))

    def __bool__(self):
        return self.passed

    def exec(self, inputs: np.ndarray, outputs: np.ndarray):

        n_samples = inputs.shape[0]
        n_input_dims = inputs.shape[1]
        n_output_dims = outputs.shape[1]

        variances = np.zeros((n_samples, n_output_dims))
        model_outs = np.zeros((n_samples, n_output_dims))

        for i in range(n_samples):

            # Mask out the sample to be predicted
            mask = (np.arange(n_samples) != i)
            train_inputs = inputs[mask]
            train_outputs = outputs[mask]

            # Train the emulator on the reduced set
            self.model.train(train_inputs, train_outputs)

            # Fix numpy chopping the bloody dimensions off silently again!!!!!
            em_in = inputs[i].reshape(1, n_input_dims)

            # Predict the missing point
            model_outs[i], variances[i] = self.model.evaluate(em_in)

        delta = np.multiply(self.sigmas, np.sqrt(variances))
        upper = np.add(outputs, delta)
        lower = np.subtract(outputs, delta)

        self.passed = not (np.any(model_outs > upper) or np.any(model_outs < lower))

        too_high = model_outs > upper
        too_low = model_outs < lower

        rate = (len(too_high) + len(too_low)) / n_samples
        if rate > self._critical_failure_rate:
            print("Diagnostic failed")

        return self

    def plot_report(self):
        pass
