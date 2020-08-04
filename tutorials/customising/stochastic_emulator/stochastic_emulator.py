#
# This example shows a two-stage stochastic emulator
#
import numpy as np
import matplotlib.pyplot as plt
from math import erf
from scipy.special import digamma, polygamma, logit
from scipy.stats import gamma, poisson, chi2
from sklearn.cluster import KMeans
from typing import List
import pandas as pd
import xehm as hm
import hdbscan as hd


#
# Poisson mixtures added together
#
class PoissonMixture(hm.Distribution):
    def __init__(self, n_mixtures=2):
        super().__init__(1, np.asarray([0, np.inf]).reshape(1, 2))
        self.num_models = n_mixtures
        self.parameters = []

    def assign(self, data):
        ints = np.asarray([int(x) for x in data], dtype=np.int).reshape(-1, 1)

        if ints.shape[0] == 0:
            self.parameters = [0] * self.num_models
            return self

        if ints.shape[0] < self.num_models:
            remainder = self.num_models % ints.shape[0]
            if remainder == 0:
                # If there is a perfect clean ratio of samples to required, then duplicate
                sources = [ints] * int(self.num_models / ints.shape[0])
            else:
                # If not, perhaps sample from biggest / smallest cluster??
                # TODO: Find a strategy here
                raise NotImplementedError("Partial resamping to be implemented")
        else:
            # Cluster out each section
            clusters = KMeans(n_clusters=self.num_models).fit(ints)
            sources = [ints[clusters.labels_ == k] for k in range(self.num_models)]

        p_list = [poisson_mle(s) for s in sources]
        self.parameters = p_list
        return self

    def pmf(self, x):
        return np.sum([poisson.pmf(x, mu=p) * (1.0 / self.num_models) for p in self.parameters])

    def expectation(self):
        return sum(p * (1.0 / self.num_models) for p in self.parameters)

    def variance(self):
        return sum(p * (1.0 / self.num_models) for p in self.parameters)

    def confidence_bound(self, sigmas: float = 2.0):
        delta = sigmas * np.sqrt(self.variance())
        return self.expectation() - delta, self.expectation() + delta


#
# Fit a Gamma to data using maximum likelihood estimation
#
def gamma_mle(data):
    sample_mean = np.mean(data)
    log_data = np.log(data)
    log_mean = np.mean(log_data)
    s = np.log(sample_mean) - log_mean

    # If we end up with only a singular non-zero category then s will be zero
    if np.isclose(s, 0.0):
        # If s is zero, then position the distribution roughly over the non-zero sample location
        alpha = data[0]
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
    return alpha, theta


def poisson_mle(data):
    return np.mean(data)


class GammaMixture(hm.Distribution):
    def __init__(self, n_mixtures: int = 2, truncate: int = None):
        if truncate is None:
            upper = np.inf
            self.truncated = False
            self.adjustment = None
        else:
            upper = truncate
            self.truncated = True
            self.adjustment = 1.0

        super().__init__(1, np.asarray([0, upper]).reshape(1, 2))
        self.num_models = n_mixtures
        self.all_zero = True
        self.p_zero = 0.0
        self.gamma_parameters = []

    # P(X == x) is defined as follows:
    #   - p for when x is 0
    #   - mixture of Gamma models when x is non-zero
    #
    def pmf(self, x):
        return np.where(x == 0, 1.0 - self.p_zero, np.sum([gamma.pdf(x, a=p[0], scale=p[1])
                        * self.p_zero * (1.0 / self.num_models) for p in self.gamma_parameters], axis=0))

    def assign(self, data, plot = False):

        # Smash into integers and filter out zeros
        ints = np.asarray([int(x) for x in data], dtype=np.int)
        non_zero = ints[ints != 0].reshape(-1, 1)

        # Use bayesian conjugation to acquire a variance - although an average will suffice
        _, b_mode, b_var, _, _ = hm.utils.bernoulli_beta(ints.tolist())
        if non_zero.shape[0] == 0:
            # All zero samples might come through here
            self._all_zero = True
            self.p_zero = 0.0
            self.gamma_parameters = []
            return self

        # It is also possible that not enough samples are non-zero to fit the mixture
        if non_zero.shape[0] < self.num_models:
            remainder = self.num_models % non_zero.shape[0]
            if remainder == 0:
                # If there is a perfect clean ratio of samples to required, then duplicate
                sources = [non_zero] * int(self.num_models / non_zero.shape[0])
            else:
                # If not, perhaps sample from biggest / smallest cluster??
                # TODO: Find a strategy here
                raise NotImplementedError("Partial resamping to be implemented")
        else:
            # Cluster out each gamma
            clusters = KMeans(n_clusters=self.num_models).fit(non_zero)
            sources = [non_zero[clusters.labels_ == k] for k in range(self.num_models)]

        p_list = [gamma_mle(s) for s in sources]

        # Normalise the PMF and correct for the Gamma truncation
        # Just in case the samples are not integers, find the maximum extents
        if self.truncated:
            x_domain = list(range(1, self.support_limits[0, 1] + 1))
            total_gamma = 0.0
            for param_set in p_list:

                # Calculate all of the mass used within the gamma component
                gamma_p_values = [gamma.pdf(x, a=param_set[0], scale=param_set[1]) for x in x_domain]
                gamma_density = np.sum(gamma_p_values) * (1.0 / self.num_models)
                total_gamma += gamma_density

            self.adjustment = 1.0 / total_gamma

        self.all_zero = False
        self.p_zero = b_mode
        self.gamma_parameters = p_list

        if not plot:
            return self

        # Ignore the truncation for now - just plot normal curves
        x_domain = np.arange(start=0, stop=data.max() + 1)
        y = self.pmf(x_domain)

        # Plotting
        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(111)
        ax = plot_samples(ax, data.squeeze().tolist(), f"Output samples")
        ax.step(x_domain, y, "-rx")
        f.show()

        return self

    def expectation(self):
        if self.truncated:
            gamma_x_range = np.arange(start=1, stop=self.support_limits[0, 1] + 1)
            mix_component = np.sum(np.multiply(gamma_x_range, self.pmf(gamma_x_range)))
        else:
            mix_component = np.sum([p[0] * p[1] for p in self.gamma_parameters])
        return self.p_zero * mix_component

    # NOTE: Each mixture component is independent so cov(mix a, mix b) = 0 for all a and all b
    def variance(self):
        if self.truncated:
            x_domain = np.arange(start=1, stop=self.support_limits[0, 1] + 1)
            e_x2 = np.sum(np.multiply(np.multiply(x_domain, x_domain), self.pmf(x_domain)))
            ex_2 = np.sum(np.multiply(x_domain, self.pmf(x_domain)))
            ex_2 = np.multiply(ex_2, ex_2)
            return e_x2 - ex_2
        else:
            b_mean = self.p_zero
            b_var = self.p_zero * (1.0 - self.p_zero)

            # Gamma bit
            g_var = np.sum([p[0] * (p[1] ** 2) for p in self.gamma_parameters])
            g_mean = np.sum([p[0] * p[1] for p in self.gamma_parameters])

            # We assume that the Bernoulli and Gamma components are independent
            return ((b_mean ** 2) * g_var) + ((g_mean ** 2) * b_var) + (b_var * g_var)

    def confidence_bound(self, sigmas: float = 2.0):
        delta = sigmas * np.sqrt(self.variance())
        return self.expectation() - delta, self.expectation() + delta

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

    # Return P(X == x) - Non-truncated
    def probability2(self, x: int):
        return np.where(x == 0, 1.0 - self.binomial_theta, gamma.pdf(x, a=self.gamma_alpha, scale=self.gamma_theta)
                        * self.binomial_theta)

    # Make the full P(X == x) listing for all X in [0, max(X)]
    def make_pmf(self):
        x_domain = list(range(self.max_support + 1))
        return [self.probability(x) for x in x_domain]

    def make_pmf2(self):
        x_domain = list(range(self.max_support + 1))
        return [self.probability2(x) for x in x_domain]

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

        alpha, theta = gamma_mle(non_zero)

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
            return self

        # Plotting
        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(111)
        ax = plot_samples(ax, data, f"Output samples")
        ax.step([0] + x_domain, self.make_pmf(), "-rx")
        f.show()
        return self

    def get_parameters(self):
        return self.binomial_theta, self.gamma_alpha, self.gamma_theta

    # draw samples using the discrete pmf method, alternatively could use a binomial followed by gamma choice
    def draw_samples(self, n_samples: int):
        elements = list(range(self.max_support + 1))
        pmf = self.make_pmf()
        return np.random.choice(elements, size=(n_samples, 1), p=pmf)

    def draw_samples2(self, n_samples: int):
        elements = list(range(self.max_support + 1))
        pmf = self.make_pmf2()
        return np.random.choice(elements, size=(n_samples, 1), p=pmf)


# Import the output data from metawards
def import_output(f_name: str) -> dict:
    tables = {}
    data: pd.DataFrame = pd.read_csv(f_name)
    out_names = data["output"].unique()
    for name in out_names:
        samples = data.loc[data['output'] == name]
        hprev = np.round(samples["Hprev"].to_numpy() * 7).astype(np.int).reshape(-1, 1)
        cprev = np.round(samples["Cprev"].to_numpy() * 7).astype(np.int).reshape(-1, 1)
        death = np.round(samples["Deaths"].to_numpy() * 7).astype(np.int).reshape(-1, 1)
        tables[name] = {"H7": hprev, "C7": cprev, "D7": death}
    return tables


# Plots a scatter of the counts over the domain of the samples
def plot_samples(axes, samples: List[int], title: str):
    x_array = list(dict.fromkeys(samples))
    y_array = [samples.count(x) / len(samples) for x in x_array]
    axes.scatter(x_array, y_array, marker="+", c="k")
    axes.set(title=title, xlabel=f"Hprev", ylabel=f"$P(Hprev)$", alpha=0.5)
    return axes


def plot_comparison(original, model, draw, id, model2=None, show=False):
    f = plt.figure(figsize=(12, 6))
    ax_left = f.add_subplot(121)
    ax_right = f.add_subplot(122)
    plot_samples(ax_left, original.squeeze().tolist(), f"Original distribution {id}")
    plot_samples(ax_right, draw.squeeze().tolist(), f"Predicted sample set {id}")
    ax_left.set_ylim(bottom=0)
    ax_left.set_xlim(left=0)
    ax_right.step(range(len(model)), model, "-rx", alpha=0.5)
    ax_right.step(range(len(model2)), model2, "-bx", alpha=0.5)
    ax_right.set_xlim(ax_left.get_xlim())
    ax_right.set_ylim(bottom=0)
    if show:
        f.show()
    f.savefig(f"C:\\mw\\figure{id}.png")


def create_emulator_table(design_file: str, tables: dict) -> dict:
    design: pd.Dataframe = pd.read_csv(design_file)
    for key in tables:
        inputs = design.loc[design["output"] == key]
        values = list(inputs.drop(columns=["output", "repeats"]).to_numpy().squeeze())
        tables[key] = values + tables[key]
    return tables


def diagnose_model(model, data):
    fails = 0
    sigma_width = 2
    for index, s in enumerate(data):
        mask = (np.arange(len(data)) != index)
        fit_samples = data[mask]
        predict = model.assign(fit_samples).confidence_bound(sigmas=sigma_width)
        observed = model.pmf(s)
        expected = np.count_nonzero(data == s) / len(data)
        c_sq = ((observed - expected) ** 2) / expected
        r = chi2.sf(c_sq, 0)
        if s < predict[0] or s > predict[1]:
            fails += 1
    f_rate = fails / len(data)
    critical_failure_rate = 1.0 - erf(sigma_width / np.sqrt(2.0))
    return not f_rate > critical_failure_rate, fails


def main():
    tables = import_output("w2139_week20.csv")
    stochastic_parameters = {}

    diag = {}
    n1_fails = 0
    n1_counts = 0
    n2_fails = 0
    n2_counts = 0
    n3_fails = 0
    n3_counts = 0
    for i, table in enumerate(tables):
        print(f"Modelling run {i + 1} of {len(tables)}")

        # Reference each sample set
        data = tables[table]
        hprev_data = data["H7"]
        cprev_data = data["C7"]
        death_data = data["D7"]

        # Attempt 1: Bernoulli-Gamma
        model_one = GammaMixture(n_mixtures=1, truncate=np.max(hprev_data))
        test_one, fails_one = diagnose_model(model_one, hprev_data)

        # Try to classify space
        if model_one.p_zero != 0:
            h = hd.HDBSCAN(min_cluster_size=2, min_samples=2).fit(hprev_data)
            cluster_count = max(h.labels_)
            if cluster_count == 1:
                h = hd.HDBSCAN(min_cluster_size=2, min_samples=1).fit(hprev_data)
                cluster_count = max(h.labels_)
            try:
                x = hm.clustering.XMeans().assign(hprev_data)
                x_count = x.n_groups
            except:
                # This is a bloody nightmare, if this throws one of the thousands of random exceptions
                # then we have no idea what to do, it might as well die
                x_count = 0
        else:
            cluster_count = 0
            x_count = 0

        if not test_one:
            print(f"H7 set {i + 1} Failed diagnostic: {fails_one} failures")
            diag[i + 1] = {"model_one": False, "clusterh": cluster_count, "clusterx": x_count}
        else:
            print(f"H7 set {i + 1} Passed diagnostic: {fails_one} failures")
            diag[i + 1] = {"model_one": True, "clusterh": cluster_count, "clusterx": x_count}

        # Attempt 2: Mixtures
        model_two = GammaMixture(n_mixtures=2, truncate=np.max(hprev_data))
        test_two, fails_two = diagnose_model(model_two, hprev_data)
        model_two.assign(hprev_data, plot = True)
        diag[i + 1]["model_two"] = test_two

        #model_three = PoissonMixture(n_mixtures=2)
        #test_three, fails_three = diagnose_model(model_three, hprev_data)
        #diag[i + 1]["model_three"] = test_three

        if not test_one:
            n1_fails += 1
        if not test_two:
            n2_fails += 1
        #if not test_three:
        #    n3_fails += 1

        n1_counts += fails_one
        n2_counts += fails_two
        #n3_counts += fails_three

        print(f"Done iteration {i + 1}. F1 {n1_fails} - F2 {n2_fails} - F3 {n3_fails}")
        print(f"Done iteration {i + 1}. F1 {n1_counts} - F2 {n2_counts} - F3 {n3_counts}")

        #feature_dist = ddist()
        #feature_dist.calibrate(hprev_data)

        #pmf = feature_dist.make_pmf()
        #pmf2 = feature_dist.make_pmf2()
        #draws = feature_dist.draw_samples(1000)
        #plot_comparison(hprev_data, pmf, draws, i + 1, pmf2)

        #params = list(feature_dist.get_parameters())
        #stochastic_parameters[table] = params
        #m.set_params(tuple(params))
        #print(f"E(x): {m.expectation()}, V(x): {m.variance()}, Conf(x): {m.confidence_bound()}")

        #s = m.sample(20)
        #f = plt.figure(figsize=(14, 7))
        #binwidth = 1
        #bins = np.arange(0, max(samples) + binwidth, binwidth)
        #ax = f.add_subplot(121)
        #ax2 = f.add_subplot(122)
        #ax2.hist(s, bins=bins, edgecolor='black', linewidth=1.0)
        #ax2.set(title="Simulated draws from model", xlabel="Hprev'", ylabel="Relative frequency")
        #ax.hist(samples, bins=bins, edgecolor='black', linewidth=1.0)
        #ax.set(title="Original samples", xlabel="Hprev'", ylabel="Relative frequency")
        #f.show()

        #ref_in = list(dict.fromkeys(samples))
        #ref_out = [samples.count(x) / len(samples) for x in ref_in]
        #predict = [pmf[x] for x in ref_in]
        #upper = []
        #lower = []
        #hm.graphics.plot_diagnostic_report(reference_inputs=ref_in, reference_outputs=ref_out, emulator_means=predict,
        #                                   emulator_upper=upper, emulator_lower=lower)

    emulator_table = create_emulator_table("design.csv", stochastic_parameters)

    h_ = []
    x_ = []
    for d in diag:
        result = diag[d]["bool"]
        if not result:
            h_.append(diag[d]["clusterh"])
            x_.append(diag[d]["clusterx"])

    print(f"MAX H: {max(h_)}, AVG H: {np.mean(h_)}")
    print(f"MAX X: {max(x_)}, AVG X: {np.mean(x_)}")

    # Try some extremely basic analyses

    # Try matching input 1 through 16 to the Bernoulli-Gamma Hprev model bernoulli parameter
    g_alpha = hm.graphics.squared_panel(16)
    g_theta = hm.graphics.squared_panel(16)
    for k in range(16):
        inputs = np.zeros((len(emulator_table), 1))
        outputs = np.zeros((len(emulator_table), 1))
        log_gamma_alpha = np.zeros((len(emulator_table), 1))
        log_gamma_theta = np.zeros((len(emulator_table), 1))
        for i, entry in enumerate(emulator_table):
            line = emulator_table[entry]
            inputs[i] = logit(np.asarray(line[k:k+1]).reshape(1, -1))
            outputs[i] = np.log(np.asarray(line[17:18]).reshape(1, -1))
            # This gets the logit(x) v log(alpha) MLE
            log_gamma_alpha[i] = np.log(np.asarray(line[17:18]).reshape(1, -1))
            log_gamma_theta[i] = np.log(np.asarray(line[18:19]).reshape(1, -1))

        lga_u = (np.mean(inputs), np.mean(log_gamma_alpha))
        print(f"Mean for model {k} gamma alpha = {lga_u}")

        # Make the E matrix
        big_x = np.c_[inputs, log_gamma_alpha]
        lga_e = None

        g_alpha.axes[k].scatter(inputs, log_gamma_alpha, marker="+", c="k")
        g_alpha.axes[k].set(title=f"Hyper-parameter plot {k + 1}", xlabel=f"logit($x_{k+1}$)",
                            ylabel=r"$ln|\alpha|$")
        g_theta.axes[k].scatter(inputs, log_gamma_theta)
        g_theta.axes[k].set(title=f"Hyper-parameter plot {k + 1}", xlabel=f"logit($x_{k + 1}$)",
                            ylabel=r"$ln|\theta|$")
    g_alpha.plot()
    g_theta.plot()

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
