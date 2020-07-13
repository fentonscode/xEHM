from numpy import ndarray, asarray, c_
from ..graphics import squared_panel
# Diagnostic reports

__all__ = ["plot_diagnostic_report"]


def plot_diagnostic_report(reference_inputs: ndarray, reference_outputs: ndarray, emulator_means: ndarray,
                           emulator_lower: ndarray, emulator_upper: ndarray):

    n_input_dimensions = reference_inputs.shape[1]
    graph = squared_panel(n_input_dimensions)
    for k in range(n_input_dimensions):

        x = reference_inputs[:, k].reshape(-1, 1)
        y1 = reference_outputs
        y3 = emulator_lower
        y4 = emulator_upper

        #graph.axes[k].scatter(x, y1, color="k", marker="+")
        ye = c_[y3.reshape(-1, ), y4.reshape(-1, )].transpose()
        graph.axes[k].errorbar(x=x, y=y1, yerr=ye, fmt="k+")

        y2 = emulator_means
        passes = asarray([(a < b or a > c) for a, b, c in zip(y2, y3, y4)])

        graph.axes[k].scatter(x[passes], y2[passes], color="g", marker="o")
        graph.axes[k].scatter(x[~passes], y2[~passes], color="r", marker="o")

        # FIXME: Put variable limits back in
        graph.axes[k].set_xlim(0.0, 1.0)
        graph.axes[k].set_ylim(bottom=0.0)

    graph.plot()