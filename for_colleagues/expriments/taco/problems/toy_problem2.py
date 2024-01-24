# Author: Yassine Laguel
# License: GNU GPL V3

import numpy as np
from numba.experimental import jitclass
from numba import int32, float64, boolean


spec = [
    ('nb_rv', int32),               # an array field
    ('constrained', boolean),
    ('data', float64[:, :, :])               # an array field
]


@jitclass(spec)
class ToyProblem2:

    def __init__(self, data, nb_rv):

        self.constrained = False

        self.nb_rv = nb_rv
        self.data = data

    def objective_func(self, x):
        return -1.0 * np.sum(np.abs(x))

    def objective_grad(self,x):
        return -1.0 * np.sign(x)

    def constraint_func(self, x, xi):
        squared_x = np.square(x)
        vec = np.zeros(self.nb_rv, dtype=np.float64)
        for ii in range(self.nb_rv):
            vec[ii] = np.dot(np.square(xi[ii]), squared_x)
        return max(vec) - 100

    def constraint_grad(self, x, xi):
        squared_x = np.square(x)
        vec = np.zeros(self.nb_rv, dtype=np.float64)
        for ii in range(self.nb_rv):
            vec[ii] = np.dot(np.square(xi[ii]), squared_x)
        i_max = np.argmax(vec)
        return np.multiply(2*x, np.square(xi[i_max]))
