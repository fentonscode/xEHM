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


def import_data(f_name: str) -> dict:
    tables = {}
    data: pd.DataFrame = pd.read_csv(f_name)
    out_names = data["output"].unique()
    for name in out_names:
        samples = data.loc[data['output'] == name]
        hprev = list(samples["Hprev"])
        fixed = [int(np.round(x * 7)) for x in hprev]
        tables[name] = fixed
    return tables


def create_emulator_table(design_file: str, tables: dict) -> dict:
    design: pd.Dataframe = pd.read_csv(design_file)
    for key in tables:
        inputs = design.loc[design["output"] == key]
        values = list(inputs.drop(columns=["output", "repeats"]).to_numpy().squeeze())
        tables[key] = values + tables[key]
    return tables


def gamma_mle(data: List[int]):
    non_zero = [d for d in data if d is not 0]
    # Binomial part
    r = len(non_zero) / len(data)

    if not non_zero:
        print(f"Data sample is all zero!")
        return (0.0, 0.0, 0.0)

    # Gamma part
    sample_mean = np.mean(non_zero)
    log_data = np.log(non_zero)
    log_mean = np.mean(log_data)
    s = np.log(sample_mean) - log_mean
    if np.isclose(s, 0.0):
        s = 1.0
    k_initial = (3.0 - s + np.sqrt(((s - 3.0) ** 2) + (24.0 * s))) / (12.0 * s)
    k = k_initial
    for x in range(100):
        kl = np.log(k)
        dg = digamma(k)
        num = kl - dg - s
        den = (1.0 / k) - polygamma(1, k)
        k = k - (num / den)
    t = sample_mean / k


    #int_max = int(np.ceil(np.max(data)))
    #x = list(range(int_max + 1))
    #y = x.copy()
    #y[0] = r
    #y[1:] = [gamma.pdf(P, a=k, scale=t) for P in x[1:]]

    #f = plt.figure(figsize=(10, 10))
    #ax = f.add_subplot(111)
    #ax.hist(data, bins=int(len(data)), density=True)
    #ax.plot(x[1:], y[1:], "--ko")
    #f.show()

    return k, t, r

def sample(k, t, r, title_str, max_count):
    s = np.random.binomial(n=1, p=r, size=1000)
    nz = s[s > 0]

    # Only draw up to a maximum value
    s2 = []
    i = 0
    while i != nz.shape[0]:
        draw = np.random.gamma(shape=k, scale=t, size=1)
        if draw <= max_count:
            s2.append(draw)
            i += 1
    z = s[s == 0]
    s4 = np.asarray(s2).reshape(-1,)
    s3 = np.r_[z, s4]

    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(111)
    ax.hist(s3, bins=int(len(s3) / 10), density=True)

    if len(s2) != 0:
        int_max = int(np.ceil(np.max(s2)))
        x = list(range(int_max + 1))
        y = x.copy()
        y[0] = r
        y[1:] = [gamma.pdf(p, a=k, scale=t) for p in x[1:]]
        ax.plot(x[1:], y[1:], "--ko")

    ax.set(title=f"{title_str}")
    f.show()

    return s3

def main():

    tables = import_data("w2139_week20.csv")
    stochastic_parameters = {}

    for i, table in enumerate(tables):
        k, t, r = gamma_mle(tables[table])
        stochastic_parameters[table] = [k, t, r]
        print(f"Gamma hurdle model for table {i + 1}: alpha = {k}, theta = {t}, binomial = {r}")
        #samples = sample(k, t, r, f"Sampler output {i + 1}", np.max(table))
        #print(f"Gamma hurdle sampled: {k} : {t} : {r}")

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
                       reference_outputs=outputs)

    default_ls = [1.0] * 16
    kernel = gp.kernels.SquaredExponential(lengthscales=default_ls)
    model = gp.models.GPR(mean_function=None, kernel=kernel, data=(inputs, outputs))
    model.likelihood.variance.assign(1.0)
    opt = gp.optimizers.Scipy()
    opt_logs = opt.minimize(model.training_loss, model.trainable_variables,
                            options=dict(maxiter=100))
    print("Trained emulator")
    pass


if __name__ == '__main__':
    main()