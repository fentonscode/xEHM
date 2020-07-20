from numpy import ndarray, asarray, c_
from ..graphics import squared_panel
# Diagnostic reports

__all__ = ["plot_diagnostic_report"]


# FIXME: What do we do for multi-output models?
# At the moment the diagnostics will only report the first output
#   - TODO: Change this to an index parameter at least, so that we can report any dimension, one at a time
def plot_diagnostic_report(reference_inputs: ndarray, reference_outputs: ndarray, emulator_means: ndarray,
                           emulator_lower: ndarray, emulator_upper: ndarray, selector: ndarray):

    n_input_dimensions = reference_inputs.shape[1]
    n_means = emulator_means.shape[1]

    for n in range(n_means):
        graph = squared_panel(n_input_dimensions)
        for k in range(n_input_dimensions):

            x = reference_inputs[:, k].reshape(-1, 1)
            y1 = reference_outputs[:, n]
            y3 = emulator_lower[:, n]
            y4 = emulator_upper[:, n]
            y2 = emulator_means[:, n]

            #graph.axes[k].scatter(x, y1, color="k", marker="+")

            passes = selector[:, n]
            ye = c_[y3[passes].reshape(-1, ), y4[passes].reshape(-1, )].transpose()
            graph.axes[k].errorbar(x=x[passes], y=y1[passes], yerr=ye, fmt="k+", alpha=0.4)
            graph.axes[k].scatter(x[passes], y2[passes], color="g", marker="o")
            #graph.axes[k].scatter(x[~passes], y2[~passes], color="r", marker="o")

            # FIXME: Put variable limits back in
            #graph.axes[k].set_xlim(0.0, 1.0)
            #graph.axes[k].set_ylim(bottom=0.0)

        graph.plot()
        graph = squared_panel(n_input_dimensions)
        for k in range(n_input_dimensions):
            x = reference_inputs[:, k].reshape(-1, 1)
            y1 = reference_outputs[:, n]
            y3 = emulator_lower[:, n]
            y4 = emulator_upper[:, n]

            # graph.axes[k].scatter(x, y1, color="k", marker="+")
            y2 = emulator_means[:, n]
            passes = selector[:, n]

            #graph.axes[k].scatter(x[passes], y2[passes], color="g", marker="o")
            graph.axes[k].scatter(x[~passes], y2[~passes], color="r", marker="o")

            ye = c_[y3[~passes].reshape(-1, ), y4[~passes].reshape(-1, )].transpose()

            graph.axes[k].errorbar(x=x[~passes], y=y1[~passes], yerr=ye, fmt="k+", alpha=0.4)

            # FIXME: Put variable limits back in
            # graph.axes[k].set_xlim(0.0, 1.0)
            # graph.axes[k].set_ylim(bottom=0.0)

        graph.plot()