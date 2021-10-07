"""
.. module:: oracle
   :synopsis: Module with definitions of first order oracle for the DC objective of the penalized problem

.. moduleauthor:: Yassine LAGUEL
"""

import numpy as np
# TODO : Put imports in init file and add licence everywhere
from .problems.toy_problem import ToyProblem
from numba.experimental import jitclass
from numba import int32, float64
from numba import njit


class Oracle:
    r"""Base class that instantiates a first-order oracle for the DC objective of the penalized chance constaint

        For two input oracles :math:`f` and :math:`g` given through their function and gradients, and a
        sampled dataset from a random variable :math:`\xi`, this class is an interface to compute the value
        and the gradient of the function
        :math:`x, \eta \mapsto f(x) + \mu \max(\eta,0) + \lambda \left(G(x,\eta) - \text{CVaR}_p(g(x,\xi))\right)`
        where :math:`G(x, \eta) = \eta + \frac{1}{1-p} \mathbb{E}[\max(g(x, \xi) - \eta]`

        :param problem: Instance of Problem
        :param ``np.float64`` p: Real number in [0,1]. Safety probability level
        :param ``np.float64`` pen1: Penalization parameter, must be positive.
        :param ``np.float64`` pen2: Penalization parameter, must be positive.
        :param ``np.float64`` rho: Smoothing parameter for the superquantile, must be positive.
    """

    def __init__(self, problem, p, pen1, pen2, rho):

        self.problem = problem

        self.p = p
        self.sample_size = len(problem.data)
        self.pen1 = pen1
        self.pen2 = pen2
        self.rho = rho

    ####################################################################################
    # FUNCTIONS COMPUTATIONS
    ####################################################################################

    def f(self, x):
        """Value of the DC objective
        """
        return self.f1(x) - self.f2(x)

    def f1(self, x):
        """Value of the convex part of the DC objective :math:`x, \eta \mapsto f(x) + \mu \max(\eta,0) + \lambda G(x,\eta)`
        """
        return self.objective_func(x[:-1]) + self.pen1 * max(x[-1], 0.0) + self.pen2 * self._G_func(x)

    def g1(self, x):
        """Gradient of the convex part of the DC objective :math:`x, \eta \mapsto f(x) + \mu \max(\eta,0) + \lambda G(x,\eta)`
        """
        res = self.pen2 * self._G_grad(x)
        res[-1] += (x[-1] > 0) * self.pen1
        res[:-1] += self.objective_grad(x[:-1])
        return res

    def f2(self, x):
        """Value of the concave part of the DC objective :math:`x, \eta \mapsto - {CVaR}_p(g(x,{\\xi}))`
        """
        f, _ = self._smooth_superquantile(x, self.problem.data)
        return self.pen2 * f

    def g2(self, x):
        """Gradient of the concave part of the DC objective :math:`x, \eta \mapsto - {CVaR}_p(g(x,{\\xi}))`
        """
        res = np.zeros_like(x, dtype=np.float64)
        _, g = self._smooth_superquantile(x, self.problem.data)
        res[:-1] = self.pen2 * g
        return res

    def _G_func(self, x):
        # return _dyn_G_func(x, self.constraint_func, self.problem.data, self.sample_size, self.p)
        vec_g_values = np.zeros(self.sample_size, dtype=np.float64)
        for ii in range(self.sample_size):
            vec_g_values[ii] = self.constraint_func(x[:-1], self.problem.data[ii])
        vec_g_values -= x[-1] * np.ones(self.sample_size, dtype=np.float64)
        vec_g_values = np.maximum(vec_g_values, np.zeros(self.sample_size, dtype=np.float64))
        return x[-1] + np.mean(vec_g_values) / (1.0 - self.p)

    def _G_grad(self, x):
        # return _dyn_G_grad(x, self.constraint_func, self.constraint_grad, self.problem.data, self.sample_size, self.p)
        res = np.zeros_like(x, dtype=np.float64)
        vec_func_values = np.zeros(self.sample_size, dtype=np.float64)
        for ii in range(self.sample_size):
            vec_func_values[ii] = self.constraint_func(x[:-1], self.problem.data[ii])

        indices_values_above_eta = np.where(vec_func_values > x[-1])[0]
        vec_grad_values_above_eta = np.zeros((len(indices_values_above_eta), len(x) - 1), dtype=np.float64)
        for ii in range(len(indices_values_above_eta)):
            vec_grad_values_above_eta[ii] = \
                self.constraint_grad(x[:-1], self.problem.data[indices_values_above_eta[ii]])

        indices_values_equal_eta = np.where(vec_func_values == x[-1])[0]

        res_x = np.sum(vec_grad_values_above_eta, axis=0) / (self.sample_size * (1.0 - self.p))
        res_eta = 1.0 - len(indices_values_above_eta) / (self.sample_size * (1.0 - self.p))

        if len(indices_values_equal_eta) > 0:
            vec_grad_values_equal_eta = np.zeros((len(indices_values_equal_eta), len(x) - 1), dtype=np.float64)
            for ii in range(len(indices_values_equal_eta)):
                vec_grad_values_equal_eta[ii] = \
                    self.constraint_grad(x[:-1], self.problem.data[indices_values_equal_eta[ii]])
            alpha = (self.sample_size * (1 - self.p) - len(indices_values_above_eta)) / (len(indices_values_equal_eta))
            if alpha < 0 or alpha > 1.0:
                if x[-1] >= 0:
                    alpha = 0.0
                else:
                    alpha = 1.0
            res_x = res_x + alpha * np.sum(vec_grad_values_equal_eta, axis=0) / (self.sample_size * (1.0 - self.p))
            res_eta = res_eta - alpha * len(indices_values_equal_eta) / (self.sample_size * (1.0 - self.p))

        res[:-1] = res_x
        res[-1] = res_eta
        return res

    ####################################################################################
    # SMOOTHING THE CVAR
    ####################################################################################

    def _smooth_superquantile(self, x, xi):
        # return _dyn_smooth_superquantile(x, xi, self.constraint_func, self.constraint_grad,
        #                                  self.sample_size, self.p, self.rho)
        n = self.sample_size

        simplex_center = 1.0 / n * np.ones(n, dtype=np.float64)

        point_to_project = np.zeros(n, dtype=np.float64)
        for i in range(n):
            point_to_project[i] = self.constraint_func(x[:-1], xi[i])

        q_mu = self._projection(point_to_project)

        f = np.dot(np.ascontiguousarray(point_to_project), np.ascontiguousarray(q_mu)) \
            - self.rho * np.linalg.norm(q_mu - simplex_center) ** 2

        jacobian_l = np.zeros((n, len(x)-1), dtype=np.float64)
        for i in range(n):
            jacobian_l[i] = self.constraint_grad(x[:-1], xi[i])

        g = np.transpose(jacobian_l).dot(q_mu)
        return f, g

    def _projection(self, u):
        return _fast_projection(u, self.p, self.rho)

    def _find_lmbda(self, v, sorted_index):
        return _fast_find_lmbda(v, sorted_index, self.p, self.rho)

    def _theta_prime(self, lmbda, v, sorted_index):
        return _fast_theta_prime(lmbda, v, sorted_index, self.p, self.rho)

    ####################################################################################
    # AUXILIARY FUNCTIONS
    ####################################################################################

    def right_eta(self, x):
        vec_constraint_values = np.zeros(self.sample_size, dtype=np.float64)
        for ii in range(self.sample_size):
            vec_constraint_values[ii] = self.constraint_func(x[:-1], self.problem.data[ii])
        return _quantile(self.p, vec_constraint_values)

    # Used in Bundle Methods
    def f1_ncc_part(self, x):
        return self.objective_func(x[:-1]) + self.pen1 * max(x[-1], 0.0)

    def g1_ncc_part(self, x):
        res = np.zeros_like(x, dtype=np.float64)
        if x[-1] > 0:
            res[-1] += self.pen1
        res[:-1] = res[:-1] + self.objective_grad(x[:-1])
        return res

    def f1_cc_part(self, x):
        return self.pen2 * self._G_func(x)

    def g1_cc_part(self, x):
        res = self.pen2 * self._G_grad(x)
        return res

    def first_order_oracle_f2(self, x):

        gradient = np.zeros_like(x, dtype=np.float64)
        func, g = self._smooth_superquantile(x, self.problem.data)
        gradient[:-1] = self.pen2 * g
        func *= self.pen2
        return func, gradient

    def objective_func(self, x):
        return self.problem.objective_func(x)

    def objective_grad(self, x):
        return self.problem.objective_grad(x)

    def constraint_func(self, x, z):
        return self.problem.constraint_func(x, z)

    def constraint_grad(self, x, z):
        return self.problem.constraint_grad(x, z)


# ####################################################################################
# # Conditionally Fast Routines
# ####################################################################################
#
#
# class DynamicJit(object):
#     def __init__(self, f):
#         self.f = f
#
#     def __call__(self, *args):
#         return self.f(*args)
#
#     def jit_this(self):
#         self.f = njit(self.f)
#
# @njit
# def _dyn_G_func(x, constraint_func, data, sample_size, p):
#     vec_g_values = np.zeros(sample_size, dtype=np.float64)
#     for ii in range(sample_size):
#         vec_g_values[ii] = constraint_func(x[:-1], data[ii])
#     vec_g_values -= x[-1] * np.ones(sample_size, dtype=np.float64)
#     vec_g_values = np.maximum(vec_g_values, np.zeros(sample_size, dtype=np.float64))
#     return x[-1] + np.mean(vec_g_values) / (1.0 - p)
#
# @njit
# def _dyn_G_grad(x, constraint_func, constraint_grad, data, sample_size, p):
#     res = np.zeros_like(x, dtype=np.float64)
#     vec_func_values = np.zeros(sample_size, dtype=np.float64)
#     for ii in range(sample_size):
#         vec_func_values[ii] = constraint_func(x[:-1], data[ii])
#
#     indices_values_above_eta = np.where(vec_func_values > x[-1])[0]
#     vec_grad_values_above_eta = np.zeros((len(indices_values_above_eta), len(x) - 1), dtype=np.float64)
#     for ii in range(len(indices_values_above_eta)):
#         vec_grad_values_above_eta[ii] = \
#             constraint_grad(x[:-1], data[indices_values_above_eta[ii]])
#
#     indices_values_equal_eta = np.where(vec_func_values == x[-1])[0]
#
#     res_x = np.sum(vec_grad_values_above_eta, axis=0) / (sample_size * (1.0 - p))
#     res_eta = 1.0 - len(indices_values_above_eta) / (sample_size * (1.0 - p))
#
#     if len(indices_values_equal_eta) > 0:
#         vec_grad_values_equal_eta = np.zeros((len(indices_values_equal_eta), len(x) - 1), dtype=np.float64)
#         for ii in range(len(indices_values_equal_eta)):
#             vec_grad_values_equal_eta[ii] = \
#                 constraint_grad(x[:-1], data[indices_values_equal_eta[ii]])
#         alpha = (sample_size * (1 - p) - len(indices_values_above_eta)) / (len(indices_values_equal_eta))
#         if alpha < 0 or alpha > 1.0:
#             if x[-1] >= 0:
#                 alpha = 0.0
#             else:
#                 alpha = 1.0
#         res_x = res_x + alpha * np.sum(vec_grad_values_equal_eta, axis=0) / (sample_size * (1.0 - p))
#         res_eta = res_eta - alpha * len(indices_values_equal_eta) / (sample_size * (1.0 - p))
#
#     res[:-1] = res_x
#     res[-1] = res_eta
#     return res
#
# @njit
# def _dyn_smooth_superquantile(x, xi, constraint_func, constraint_grad, sample_size, p, rho):
#
#     n = sample_size
#
#     simplex_center = 1.0 / n * np.ones(n, dtype=np.float64)
#
#     point_to_project = np.zeros(n, dtype=np.float64)
#     for i in range(n):
#         point_to_project[i] = constraint_func(x[:-1], xi[i])
#
#     q_mu = _fast_projection(point_to_project, p, rho)
#
#     f = np.dot(np.ascontiguousarray(point_to_project), np.ascontiguousarray(q_mu)) \
#         - rho * np.linalg.norm(q_mu - simplex_center) ** 2
#
#     jacobian_l = np.zeros((n, len(x)-1), dtype=np.float64)
#     for i in range(n):
#         jacobian_l[i] = constraint_grad(x[:-1], xi[i])
#
#     g = np.transpose(jacobian_l).dot(q_mu)
#     return f, g

spec = [
    ('problem', ToyProblem.class_type.instance_type),               # a jitclass
    ('sample_size', int32),               # a simple scalar field
    ('p', float64),               # a simple scalar field
    ('pen1', float64),               # a simple scalar field
    ('pen2', float64),          # a simple scalar field
    ('rho', float64),          # a simple scalar field
]


@njit
def _fast_projection(u, p, rho):

    n = len(u)
    c = 1.0 / (n * (1 - p))
    v = u + (2.0 * rho / n) * np.ones(n, dtype=np.float64)

    # sorts the coordinate of v
    sorted_index = np.argsort(v)

    # finds the zero of the function theta prime
    lmbda = _fast_find_lmbda(v, sorted_index, p, rho)

    # instantiate the output
    res = np.zeros(n, dtype=np.float64)

    # fills the coordinate of the output
    counter = n - 1
    while counter >= 0:
        if lmbda > v[sorted_index[counter]]:
            break
        elif lmbda > v[sorted_index[counter]] - 2 * rho * c:
            res[sorted_index[counter]] = (v[sorted_index[counter]] - lmbda) / (2 * rho)
        else:
            res[sorted_index[counter]] = c
        counter -= 1

    return res


####################################################################################
# FAST Routines
####################################################################################

@jitclass(spec)
class FastOracle:
    r"""Numba version of the class Oracle.

        FastOracle works with numba in full no-python mode. It must take as an input an instance of problem
        that is a jitclass.

        :param problem: Instance of Problem. Must be a numba jitclass.
        :param ``np.float64`` p: Real number in [0,1]. Safety probability level
        :param ``np.float64`` pen1: Penalization parameter, must be positive.
        :param ``np.float64`` pen2: Penalization parameter, must be positive.
        :param ``np.float64`` rho: Smoothing parameter for the superquantile, must be positive.
    """

    def __init__(self, problem, p, pen1, pen2, rho):

        self.problem = problem

        self.p = p
        self.sample_size = len(problem.data)
        self.pen1 = pen1
        self.pen2 = pen2
        self.rho = rho

    ####################################################################################
    # FUNCTIONS COMPUTATIONS
    ####################################################################################

    def f(self, x):
        """Value of the DC objective
        """
        return self.f1(x) - self.f2(x)

    def f1(self, x):
        """Value of the convex part of the DC objective
        """
        return self.objective_func(x[:-1]) + self.pen1 * max(x[-1], 0.0) + self.pen2 * self._G_func(x)

    def g1(self, x):
        """Gradient of the convex part of the DC objective
        """
        res = self.pen2 * self._G_grad(x)
        res[-1] += (x[-1] > 0) * self.pen1
        res[:-1] += self.objective_grad(x[:-1])
        return res

    def f2(self, x):
        """Value of the convave part of the DC objective
        """
        f, _ = self._smooth_superquantile(x, self.problem.data)
        return self.pen2 * f

    def g2(self, x):
        """Gradient of the convave part of the DC objective
        """
        res = np.zeros_like(x, dtype=np.float64)
        _, g = self._smooth_superquantile(x, self.problem.data)
        res[:-1] = self.pen2 * g
        return res

    def _G_func(self, x):
        # return _dyn_G_func(x, self.constraint_func, self.problem.data, self.sample_size, self.p)
        vec_g_values = np.zeros(self.sample_size, dtype=np.float64)
        for ii in range(self.sample_size):
            vec_g_values[ii] = self.constraint_func(x[:-1], self.problem.data[ii])
        vec_g_values -= x[-1] * np.ones(self.sample_size, dtype=np.float64)
        vec_g_values = np.maximum(vec_g_values, np.zeros(self.sample_size, dtype=np.float64))
        return x[-1] + np.mean(vec_g_values) / (1.0 - self.p)

    def _G_grad(self, x):
        # return _dyn_G_grad(x, self.constraint_func, self.constraint_grad, self.problem.data, self.sample_size, self.p)
        res = np.zeros_like(x, dtype=np.float64)
        vec_func_values = np.zeros(self.sample_size, dtype=np.float64)
        for ii in range(self.sample_size):
            vec_func_values[ii] = self.constraint_func(x[:-1], self.problem.data[ii])

        indices_values_above_eta = np.where(vec_func_values > x[-1])[0]
        vec_grad_values_above_eta = np.zeros((len(indices_values_above_eta), len(x) - 1), dtype=np.float64)
        for ii in range(len(indices_values_above_eta)):
            vec_grad_values_above_eta[ii] = \
                self.constraint_grad(x[:-1], self.problem.data[indices_values_above_eta[ii]])

        indices_values_equal_eta = np.where(vec_func_values == x[-1])[0]

        res_x = np.sum(vec_grad_values_above_eta, axis=0) / (self.sample_size * (1.0 - self.p))
        res_eta = 1.0 - len(indices_values_above_eta) / (self.sample_size * (1.0 - self.p))

        if len(indices_values_equal_eta) > 0:
            vec_grad_values_equal_eta = np.zeros((len(indices_values_equal_eta), len(x) - 1), dtype=np.float64)
            for ii in range(len(indices_values_equal_eta)):
                vec_grad_values_equal_eta[ii] = \
                    self.constraint_grad(x[:-1], self.problem.data[indices_values_equal_eta[ii]])
            alpha = (self.sample_size * (1 - self.p) - len(indices_values_above_eta)) / (len(indices_values_equal_eta))
            if alpha < 0 or alpha > 1.0:
                if x[-1] >= 0:
                    alpha = 0.0
                else:
                    alpha = 1.0
            res_x = res_x + alpha * np.sum(vec_grad_values_equal_eta, axis=0) / (self.sample_size * (1.0 - self.p))
            res_eta = res_eta - alpha * len(indices_values_equal_eta) / (self.sample_size * (1.0 - self.p))

        res[:-1] = res_x
        res[-1] = res_eta
        return res

    ####################################################################################
    # SMOOTHING THE CVAR
    ####################################################################################

    def _smooth_superquantile(self, x, xi):
        # return _dyn_smooth_superquantile(x, xi, self.constraint_func, self.constraint_grad,
        #                                  self.sample_size, self.p, self.rho)
        n = self.sample_size

        simplex_center = 1.0 / n * np.ones(n, dtype=np.float64)

        point_to_project = np.zeros(n, dtype=np.float64)
        for i in range(n):
            point_to_project[i] = self.constraint_func(x[:-1], xi[i])

        q_mu = self._projection(point_to_project)

        f = np.dot(np.ascontiguousarray(point_to_project), np.ascontiguousarray(q_mu)) \
            - self.rho * np.linalg.norm(q_mu - simplex_center) ** 2

        jacobian_l = np.zeros((n, len(x)-1), dtype=np.float64)
        for i in range(n):
            jacobian_l[i] = self.constraint_grad(x[:-1], xi[i])

        g = np.transpose(jacobian_l).dot(q_mu)
        return f, g

    def _projection(self, u):
        return _fast_projection(u, self.p, self.rho)

    def _find_lmbda(self, v, sorted_index):
        return _fast_find_lmbda(v, sorted_index, self.p, self.rho)

    def _theta_prime(self, lmbda, v, sorted_index):
        return _fast_theta_prime(lmbda, v, sorted_index, self.p, self.rho)

    ####################################################################################
    # AUXILIARY FUNCTIONS
    ####################################################################################

    def right_eta(self, x):
        vec_constraint_values = np.zeros(self.sample_size, dtype=np.float64)
        for ii in range(self.sample_size):
            vec_constraint_values[ii] = self.constraint_func(x[:-1], self.problem.data[ii])
        return _quantile(self.p, vec_constraint_values)

    # Used in Bundle Methods
    def f1_ncc_part(self, x):
        return self.objective_func(x[:-1]) + self.pen1 * max(x[-1], 0.0)

    def g1_ncc_part(self, x):
        res = np.zeros_like(x, dtype=np.float64)
        if x[-1] > 0:
            res[-1] += self.pen1
        res[:-1] = res[:-1] + self.objective_grad(x[:-1])
        return res

    def f1_cc_part(self, x):
        return self.pen2 * self._G_func(x)

    def g1_cc_part(self, x):
        res = self.pen2 * self._G_grad(x)
        return res

    def first_order_oracle_f2(self, x):

        gradient = np.zeros_like(x, dtype=np.float64)
        func, g = self._smooth_superquantile(x, self.problem.data)
        gradient[:-1] = self.pen2 * g
        func *= self.pen2
        return func, gradient

    def objective_func(self, x):
        return self.problem.objective_func(x)

    def objective_grad(self, x):
        return self.problem.objective_grad(x)

    def constraint_func(self, x, z):
        return self.problem.constraint_func(x, z)

    def constraint_grad(self, x, z):
        return self.problem.constraint_grad(x, z)


@njit
def _fast_find_lmbda(v, sorted_index, p, rho):

    n = len(v)
    c = np.float64(1.0 / (n * (1.0 - p)))
    set_p = np.sort(np.concatenate((v, v - 2 * rho * c * np.ones(n, dtype=np.float64))))

    def aux(a=0, b=2 * n - 1):

        m = (a + b) // 2
        while (b - a) > 1:
            if _fast_theta_prime(set_p[m], v, sorted_index, p, rho) > 0:
                b = m
            elif _fast_theta_prime(set_p[m], v, sorted_index, p, rho) < 0:
                a = m
            else:
                return set_p[m]
            m = (a + b) // 2

        res = set_p[a] - (_fast_theta_prime(set_p[a], v, sorted_index, p, rho) * (set_p[b] - set_p[a])) / (
                _fast_theta_prime(set_p[b], v, sorted_index, p, rho) - _fast_theta_prime(set_p[a],
                                                                                         v, sorted_index, p, rho))
        return res
    return aux()


@njit
def _fast_theta_prime(lmbda, v, sorted_index, p, rho):

    n = len(v)
    c = 1.0 / (n * (1.0 - p))

    res = 1.0
    counter = n - 1
    while counter >= 0:
        if lmbda >= v[sorted_index[counter]]:
            break
        elif lmbda >= v[sorted_index[counter]] - 2 * rho * c:
            res -= (v[sorted_index[counter]] - lmbda) / (2 * rho)
        else:
            res -= c
        counter -= 1
    return res


# @njit
# def _quantile(p, u):
#     v = np.sort(u)
#     if p == 0:
#         return v[0]
#     else:
#         n = len(v)
#         index = int(np.ceil(n * p)) - 1
#         return v[index]

@njit
def _quantile(p, u):
    if p == 0:
        k = 1
    else:
        k = np.ceil(p * len(u))
    return _quickselect(k, u)


@njit
def _quickselect(k, list_of_numbers):
    return _kthSmallest(list_of_numbers, k, 0, len(list_of_numbers) - 1)


@njit
def _kthSmallest(arr, k, start, end):

    pivot_index = _partition(arr, start, end)

    if pivot_index - start == k - 1:
        return arr[pivot_index]

    if pivot_index - start > k - 1:
        return _kthSmallest(arr, k, start, pivot_index - 1)

    return _kthSmallest(arr, k - pivot_index + start - 1, pivot_index + 1, end)


@njit
def _partition(arr, l, r):

    pivot = arr[r]
    i = l
    for j in range(l, r):

        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    arr[i], arr[r] = arr[r], arr[i]
    return i
