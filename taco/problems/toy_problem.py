# Author: Yassine Laguel
# License: GNU GPL V3

import numpy as np
from numba.experimental import jitclass
from numba import int32, float64


spec = [
    ('a', float64[:]),               # a simple scalar field
    ('data', float64[:, :]),               # an array field
    ('geo_a', float64[:, :]),          # an array field
]


@jitclass(spec)
class ToyProblem:

    def __init__(self, data):

        self.a = np.array([2.0, 2.0], dtype=np.float64)
        self.data = data

        r = np.sqrt(2)/2
        rot = np.array([[r, -1.0 * r], [r, r]], dtype=np.float64)
        inv_rot = np.array([[r, r], [-1.0 * r, r]], dtype=np.float64)
        mat_in = np.array([[1., 0.], [0., 10.]], dtype=np.float64)
        self.geo_a = np.dot(inv_rot, np.dot(mat_in, rot))

    def objective_func(self, x):

        return 0.5 * np.dot(x - self.a, np.dot(np.ascontiguousarray(self.geo_a), x - self.a))

    def objective_grad(self, x):

        return np.dot(self.geo_a, x - self.a)

    def constraint_func(self, x, z):
        beta = np.array([1., 1.], dtype=np.float64)
        a = np.dot(np.transpose(z), np.dot(self.mat_w(x), z))
        b = np.dot(beta, z)
        c = -1.
        res = a + b + c
        return res

    @staticmethod
    def constraint_grad(x, z):

        g0 = 2 * x[0] * z[0]**2
        g1 = 3 * np.sign(x[1] - 1.0) * (z[1] * (x[1] - 1.0))**2
        res = np.array([g0, g1])
        return res

    @staticmethod
    def mat_w(x):
        d1 = x[0] ** 2 + 0.5
        d2 = abs((x[1] - 1)) ** 3 + 0.2
        res = np.diag(np.array([d1, d2], dtype=np.float64))
        return res
