import numpy as np


# Fit a quadratic to two scalar x/y pairs and a derivative at x1/y1
def quad_construct(x1, x2, y1, y2, d1):
    m = np.array([[(x1 ** 2), x1, x1], [(x2 ** 2), x2, 1], [(2 * x1), 1, 0]])
    b = np.array([y1, y2, d1])
    return np.linalg.solve(m, b)


# Fit a cubic to two x/y pairs with a derivative at each
def cubic_construct(x1, x2, y1, y2, d1, d2):
    m1 = [(x1 ** 3), (x1 ** 2), x1, 1]
    m2 = [(x2 ** 3), (x2 ** 2), x2, 1]
    m3 = [3 * (x1 ** 2), 2 * x1, 1, 0]
    m4 = [3 * (x2 ** 2), 2 * x2, 1, 0]

    # solve() should avoid computing a direct inverse
    m = np.array([m1, m2, m3, m4])
    return np.linalg.solve(m, np.array([y1, y2, d1, d2]))


