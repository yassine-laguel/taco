# Author: Yassine Laguel
# License: GNU GPL V3

# Implementation of the toy problem from a simple linear toy problem

import numpy as np
from numba.experimental import jitclass
from numba import float64, boolean

spec = [
    ('constrained', boolean),
    ('data', float64[:, :])
]

@jitclass(spec)
class ToyProblem4:

    def __init__(self, data):

        self.constrained = False
        self.data = data

    @staticmethod
    def objective_func(x):
        return -np.sum(x)

    @staticmethod
    def objective_grad(x):
        return -1 * np.ones(2, dtype=np.float64)

    @staticmethod
    def constraint_func(x, xi):
        return xi[0] * x[0] + (xi[1]+3) * x[1] - 5

    @staticmethod
    def constraint_grad(x, xi):
        res = np.zeros(2, dtype=np.float64)
        res[1] = 3
        res += xi
        return res
