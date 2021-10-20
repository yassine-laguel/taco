# Author: Yassine Laguel
# License: GNU GPL V3

# Implementation of the toy problem from "CONVEX APPROXIMATIONS OF CHANCE CONSTRAINED PROGRAMS" by Nemirovski and Shapiro

import numpy as np
from numba.experimental import jitclass
from numba import int32, float64, boolean

spec = [
    ('nb_rv', int32),               # an array field
    ('data', float64[:, :]),               # an array field
    ('constrained', boolean),
    ('equality_constrained', boolean),
    ('inequality_constrained', boolean),
    ('G', float64[:, :]),
    ('h', float64[:]),
    # ('A', float64[:, :]),
    # ('b', float64[:])
]


@jitclass(spec)
class ToyProblem3:

    def __init__(self, data):
        self.constrained = True
        self.inequality_constrained = True
        self.equality_constrained = False
        self.data = data

        d = 65
        self.G = -1.0 * np.eye(d+1, dtype=np.float64)
        self.G[-1] = np.ones(d+1, dtype=np.float64)
        self.G[-1][-1] = 0.
        self.h = np.zeros(d+1, dtype=np.float64)
        self.h[-1] = 1.0
        # self.A = np.ones((1, d), dtype=np.float64)
        # self.A[0][-1] = 0.
        # self.b = np.ones(1)

    @staticmethod
    def objective_func(x):
        return 1.0 - x[-1]

    @staticmethod
    def objective_grad(x):
        res = np.zeros_like(x)
        res[-1] = -1.
        return res

    @staticmethod
    def constraint_func(x, xi):
        return x[-1] - np.dot(xi, x[:-1])

    @staticmethod
    def constraint_grad(x, xi):
        res = np.empty_like(x)
        res[:-1] = -1.0 * xi
        res[-1] = 1.
        return res
