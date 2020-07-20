#
# This example shows a two-stage stochastic emulator
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, polygamma
from scipy.stats import gamma
from typing import List
import pandas as pd
import gpflow as gp
import xehm as hm


# A simple 3 parameter discrete distribution to calibrate for modelling
# stochastic metawards data
class ddist():
    def __init__(self):
        self.max_support: int = 0
        self.gamma_alpha: float = 0.0
        self.gamma_theta: float = 0.0
        self.binomial_theta: float = 0.0
        self.gamma_adjustment: float = 0.0
        self._all_zero = True

    # Return P(X == x)
    def probability(self, x: int):
        if x == 0:
            return 1.0 - self.binomial_theta
        return gamma.pdf(x, a=self.gamma_alpha, scale=self.gamma_theta) * self.gamma_adjustment * self.binomial_theta

    # Make the full P(X == x) listing for all X in [0, max(X)]
    def make_pmf(self):
        x_domain = list(range(self.max_support + 1))
        return [self.probability(x) for x in x_domain]

    # Fit P(X == x) to specified samples using a binomial-Gamma model
    def calibrate(self, samples, plot: bool = False, max_x: float = None):

        # Samples should be ints, so force a conversion here
        data = [int(x) for x in samples]

        # Binomial part
        non_zero = [d for d in data if d is not 0]

        # Use bayesian conjugation to acquire a variance - although an average will suffice
        _, b_mode, b_var, _, _ = hm.utils.bernoulli_beta(data)
        if not non_zero:
            # All zero samples might come through here
            self._all_zero = True
            return

        # Gamma part
        sample_mean = np.mean(non_zero)
        log_data = np.log(non_zero)
        log_mean = np.mean(log_data)
        s = np.log(sample_mean) - log_mean

        # If we end up with only a singular non-zero category then s will be zero
        if np.isclose(s, 0.0):
            # If s is zero, then position the distribution roughly over the non-zero sample location
            alpha = non_zero[0]
        else:
            k_initial = (3.0 - s + np.sqrt(((s - 3.0) ** 2) + (24.0 * s))) / (12.0 * s)
            alpha = k_initial
            # TODO: Change this for a tolerance based loop?
            for x in range(10):
                kl = np.log(alpha)
                dg = digamma(alpha)
                num = kl - dg - s
                den = (1.0 / alpha) - polygamma(1, alpha)
                alpha = alpha - (num / den)

        theta = sample_mean / alpha

        # Normalise the PMF and correct for the Gamma truncation
        # Just in case the samples are not integers, find the maximum extents
        if max_x is None:
            max_x = int(np.ceil(np.max(samples)))
        x_domain = list(range(1, max_x + 1))

        # Calculate all of the mass used within the gamma component
        gamma_p_values = [gamma.pdf(x, a=alpha, scale=theta) for x in x_domain]
        gamma_density = np.sum(gamma_p_values)
        adjustment = 1.0 / gamma_density

        # Set attributes
        self.binomial_theta = b_mode
        self.gamma_alpha = alpha
        self.gamma_theta = theta
        self.gamma_adjustment = adjustment
        self.max_support = max_x
        self._all_zero = False

        if not plot:
            return

        # Plotting
        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(111)
        ax = plot_samples(ax, data, f"Output samples")
        ax.step([0] + x_domain, self.make_pmf(), "-rx")
        f.show()

    def get_parameters(self):
        return self.gamma_alpha, self.gamma_theta, self.binomial_theta

    # draw samples using the discrete pmf method, alternatively could use a binomial followed by gamma choice
    def draw_samples(self, n_samples: int):
        elements = list(range(self.max_support + 1))
        pmf = self.make_pmf()
        return np.random.choice(elements, size=(n_samples, 1), p=pmf)


def counts_to_frequencies(samples):
    # Samples should be ints, so force a conversion here
    data = [int(x) for x in samples]
    x_array = range(max(data))
    y_array = [samples.count(x) / len(samples) for x in x_array]


# Import the output data from metawards
def import_output(f_name: str) -> dict:
    tables = {}
    data: pd.DataFrame = pd.read_csv(f_name)
    out_names = data["output"].unique()
    for name in out_names:
        samples = data.loc[data['output'] == name]
        hprev = list(samples["Hprev"])
        fixed = [int(np.round(x * 7)) for x in hprev]
        tables[name] = fixed
    return tables


# Plots a scatter of the counts over the domain of the samples
def plot_samples(axes, samples: List[int], title: str):
    x_array = list(dict.fromkeys(samples))
    y_array = [samples.count(x) / len(samples) for x in x_array]
    axes.scatter(x_array, y_array, marker="+", c="k")
    axes.set(title=title, xlabel=f"Hprev", ylabel=f"$P(Hprev)$", alpha=0.5)
    return axes


def plot_comparison(original, model, draw, id):
    f = plt.figure(figsize=(12, 6))
    ax_left = f.add_subplot(121)
    ax_right = f.add_subplot(122)
    plot_samples(ax_left, original, f"Original distribution {id}")
    plot_samples(ax_right, draw.squeeze().tolist(), f"Predicted sample set {id}")
    ax_left.set_ylim(bottom=0)
    ax_left.set_xlim(left=0)
    ax_right.step(range(len(model)), model, "-rx", alpha=0.5)
    ax_right.set_xlim(ax_left.get_xlim())
    ax_right.set_ylim(bottom=0)
    f.show()


def create_emulator_table(design_file: str, tables: dict) -> dict:
    design: pd.Dataframe = pd.read_csv(design_file)
    for key in tables:
        inputs = design.loc[design["output"] == key]
        values = list(inputs.drop(columns=["output", "repeats"]).to_numpy().squeeze())
        tables[key] = values + tables[key]
    return tables


def main():
    tables = import_output("w2139_week20.csv")
    stochastic_parameters = {}

    for i, table in enumerate(tables):
        samples = tables[table]
        feature_dist = ddist()

        # Diagnose the model

#        # Select out the unique values of the samples and count them to determine probabilities
#        train_set = list(dict.fromkeys(samples))
#        actuals = []
#        predictions = []
#        for k in range(len(train_set)):
#            train_in = [s for i, s in enumerate(train_set) if i != k]
#            test = samples.count(train_set[k]) / len(samples)
#            actuals.append(test)
#            calibrate_samples = [s for k, s in enumerate(samples) if s in train_in]
#            feature_dist.calibrate(calibrate_samples, max_x=max(train_set))
#            model_predictions = feature_dist.make_pmf()
#            # Beware: this will go out of range on the last predictor
#            predictions.append(model_predictions[train_set[k]])
#            #print(f"Should be {test} : we got {predict}")##
#
#        fig = plt.figure(figsize=(6, 6))
#        ax = fig.add_subplot(111)
#        ax.scatter(train_set, actuals, c="k", marker="o")
#        ax.scatter(train_set, predictions, c="r", marker="x")
#        fig.show()

        feature_dist.calibrate(samples)
        pmf = feature_dist.make_pmf()
        draws = feature_dist.draw_samples(1000)
        #plot_comparison(tables[table], pmf, draws, i + 1)
        params = list(feature_dist.get_parameters())
        stochastic_parameters[table] = params

        ref_in = list(dict.fromkeys(samples))
        ref_out = [samples.count(x) / len(samples) for x in ref_in]
        predict = [pmf[x] for x in ref_in]

        upper = []
        lower = []

        #hm.graphics.plot_diagnostic_report(reference_inputs=ref_in, reference_outputs=ref_out, emulator_means=predict,
        #                                   emulator_upper=upper, emulator_lower=lower)

    emulator_table = create_emulator_table("design.csv", stochastic_parameters)

    inputs = np.zeros((len(emulator_table), 16))
    outputs = np.zeros((len(emulator_table), 3))
    for i, entry in enumerate(emulator_table):
        line = emulator_table[entry]
        inputs[i] = np.asarray(line[0:16]).reshape(1, -1)
        outputs[i] = np.asarray(line[16:]).reshape(1, -1)

    diagnostic = hm.diagnostics.leave_one_out()[0]
    emulator = hm.emulators.GaussianProcess()
    valid = diagnostic(emulator_model=emulator, reference_inputs=inputs,
                       reference_outputs=outputs, **{"plot_report": False})
    print("Trained emulator")
    pass


if __name__ == '__main__':
    main()
