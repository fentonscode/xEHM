# Leave-one-out (LOO) cross validation
# NOTE: scikit-learn has something called this, but it doesn't do what we need it to do

from ..emulators.emulator import Emulator
import numpy as np

__all__ = ["LeaveOneOut"]


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
            model_outs[i], variances[i] = emulator.evaluate(inputs[i])
            errors[i] = model_outs[i] - outputs[i]
        delta = np.multiply(2.0, np.sqrt(variances))
        upper = np.add(outputs, delta)
        lower = np.subtract(outputs, delta)
        return not (np.any(model_outs > upper) or np.any(model_outs < lower))
