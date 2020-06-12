# Leave-one-out (LOO) cross validation
# NOTE: scikit-learn has something called this, but it doesn't do what we need it to do

from ..emulators.emulator import Emulator
import numpy as np

__all__ = ["cross_validate"]


#
# Leave-one-out cross validation tests an emulators *structure* to see that it can generalise
#   - This will re-train the emulator, so to be safe we copy it
#
def cross_validate(emulator: Emulator, inputs: np.ndarray, outputs: np.ndarray) -> bool:
    n_samples = inputs.shape[0]
    errors = np.zeros((n_samples, outputs.shape[1]))
    variances = np.zeros((n_samples, outputs.shape[1]))
    predict_input = np.zeros((n_samples, inputs.shape[1]))
    predict_output = np.zeros((n_samples, outputs.shape[1]))
    model_outs = np.zeros((n_samples, outputs.shape[1]))
    for i in range(n_samples):
        mask = (np.arange(n_samples) != i)
        train_inputs = inputs[mask]
        train_outputs = outputs[mask]
        predict_input[i] = inputs[i]
        predict_output[i] = outputs[i]
        emulator.train(train_inputs, train_outputs)
        model_outs[i], variances[i] = emulator.evaluate(predict_input[i])
        errors[i] = model_outs[i] - predict_output[i]
    delta = np.multiply(2.0, np.sqrt(variances))
    upper = np.add(predict_output, delta)
    lower = np.subtract(predict_output, delta)
    return not (np.any(model_outs > upper) or np.any(model_outs < lower))
