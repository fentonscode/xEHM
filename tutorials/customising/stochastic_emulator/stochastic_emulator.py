#
# This example shows a two-stage stochastic emulator
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, polygamma
from scipy.stats import gamma
from typing import List
import pandas as pd

test_data = [137, 45, 69, 61, 121, 86, 71, 56, 59, 94, 59, 61, 78, 85, 46, 26, 63, 106, 27, 92]
test_data2 = [0, 8, 33, 7, 11, 6, 5, 2, 9, 13, 0, 14, 0, 0, 0, 0, 7, 43, 7, 0]


def import_data(f_name: str):
    data: pd.DataFrame = pd.read_csv(f_name)
    out_names = data["output"].unique()
    return out_names


def gamma_mle(data: List[int]):
    zeros = [d for d in data if d is 0]
    non_zero = [d for d in data if d is not 0]
    # Binomial part
    r = len(non_zero) / len(data)

    # Gamma part
    sample_mean = np.mean(non_zero)
    log_data = np.log(non_zero)
    log_mean = np.mean(log_data)
    s = np.log(sample_mean) - log_mean
    k_initial = (3.0 - s + np.sqrt(((s - 3.0) ** 2) + (24.0 * s))) / (12.0 * s)
    k = k_initial
    for x in range(100):
        kl = np.log(k)
        dg = digamma(k)
        num = kl - dg - s
        den = (1.0 / k) - polygamma(1, k)
        k = k - (num / den)
    t = sample_mean / k


    int_max = int(np.ceil(np.max(data)))
    x = list(range(int_max + 1))
    y = x.copy()
    y[0] = r
    y[1:] = [gamma.pdf(P, a=k, scale=t) for P in x[1:]]

    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(111)
    ax.hist(data, bins=int(len(data)), density=True)
    ax.plot(x[1:], y[1:], "--ko")
    f.show()

    return k, t, r

def sample(k, t, r):
    s = np.random.binomial(n=1, p=r, size=20)
    nz = s[s > 0]
    s2 = np.random.gamma(shape=k, scale=t, size=nz.shape[0])
    s3 = np.r_[s[s == 0], s2]

    int_max = int(np.ceil(np.max(s2)))
    x = list(range(int_max + 1))
    y = x.copy()
    y[0] = r
    y[1:] = [gamma.pdf(p, a=k, scale=t) for p in x[1:]]

    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(111)
    ax.hist(s3, bins=int(len(s3)), density=True)
    ax.plot(x[1:], y[1:], "--ko")
    f.show()

    return s3

def main():

    tables = import_data("w2139_week20.csv")

    k, t, r = gamma_mle(test_data2)
    samples = sample(k, t, r)
    print(f"Gamma: {k} : {t}")

if __name__ == '__main__':
    main()