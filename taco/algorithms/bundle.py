"""
.. module:: bundle
   :synopsis: Module with implementation of W. de Oliveira bundle algorithm

.. moduleauthor:: Yassine LAGUEL
"""

import numpy as np
from cvxopt import matrix, solvers
import time
import sys
from numba import njit


class BundleAlgorithm:
    """ Base class that combines the penalization procedure with a Bundle Method to solve
        the chance constraint problem. It is instantiated with a first-oder oracle for the DC objective and a
        dictionary of parameters. From time to time, the penalization parameters are increased to escape some
        critical point of the DC objective.

        :param oracle: An oracle object.
        :param  params: Python dictionary of parameters.
        :type params: dict
    """

    def __init__(self, oracle, params):
        self.oracle = oracle

        # Counter and Current Iterate
        self.counter = 0
        self.x = params['starting_point']

        # Bundle related Info
        self.max_size_bundle_set = params['bund_max_size_bundle_set']
        self.bundle_f1_infos_cc = np.zeros(self.max_size_bundle_set, dtype=np.float64)
        self.bundle_f1_infos_ncc = np.zeros(self.max_size_bundle_set, dtype=np.float64)
        self.bundle_g1_infos_cc = np.zeros((self.max_size_bundle_set, len(self.x)), dtype=np.float64)
        self.bundle_g1_infos_ncc = np.zeros((self.max_size_bundle_set, len(self.x)), dtype=np.float64)
        self.bundle_vectors = np.zeros((self.max_size_bundle_set, len(self.x)), dtype=np.float64)
        self.bundle_size = 0

        # Stability center
        self.is_serious_step = True
        self.stability_center = np.copy(self.x)
        f1 = self._get_new_linearization(self.x)
        f2, g2 = self.oracle.first_order_oracle_f2(self.x)
        self.f_stability = f1 - f2
        self.g2_l = g2

        # Algorithm constants
        self.nb_iterations = params['nb_iterations']
        self.kappa = params['bund_kappa']
        self.mu_low = params['bund_mu_low']
        self.mu_high = params['bund_mu_high']
        self.starting_mu = params['bund_mu_start']
        self.mu = self.starting_mu
        self.delta_tol = params['bund_delta_tol']
        self.scaling_term = params['bund_scaling_term']
        self.mu_inc = params['bund_mu_inc']
        self.mu_dec = params['bund_mu_dec']
        self.epsilon_alpha = params['bund_epsilon_alpha']

        # Restarting constants
        self.restarting_period = params['bund_restarting_period']
        self.restarting_mu = params['bund_mu_restart']
        self.restarting_penalty_slack = params['bund_restarting_penalty_slack']
        self.restarting_epsilon_eta = params['bund_restarting_epsilon_eta']
        self.restarting_factor_pen = params['pen2_factor']

        # Logging Tools
        self.verbose = None
        self.logging_dictionary = {}
        self.lst_iterates = np.zeros((self.nb_iterations, len(self.x)), dtype=np.float64)
        self.lst_times = np.zeros(self.nb_iterations, dtype=np.float64)
        self.lst_values = np.zeros(self.nb_iterations, dtype=np.float64)
        self.lst_serious_steps = np.zeros(self.nb_iterations, dtype=np.float64)

    def run(self, verbose=False, logs=False):
        """ Runs the optimization process
        :type verbose: bool
        :param verbose: If true, prints advance of the process in the console
        :return: solution of the problem
        """

        start_time = time.time()
        self.verbose = verbose

        while self.counter < self.nb_iterations:

            if logs:
                self.lst_times[self.counter] = time.time() - start_time
                self.lst_iterates[self.counter] = self.stability_center
                self.lst_values[self.counter] = self.f_stability
                if self.is_serious_step:
                    self.lst_serious_steps[self.counter] = 1.0
            if verbose:
                sys.stdout.write('%d / %d  iterations completed \r' % (self.counter+1, self.nb_iterations))
                sys.stdout.flush()

            new_x = self._solve_sub_problem()

            if np.linalg.norm(new_x - self.stability_center) <= self.delta_tol:
                if verbose:
                    print("Solution Found in " + str(self.counter) + " iterations.")
                return self.stability_center
            else:
                new_f = self.oracle.f(new_x)
                self.is_serious_step = self._eval_serious_condition(new_f, new_x)
                self._update_bundle_items(new_x)
                self._update_mu()
                self.x = new_x

            self._restart_if_needed()
            self.counter += 1

        return self.stability_center

    ####################################################################################
    # Solve SubProblem
    ####################################################################################

    def _solve_sub_problem(self):

        Q = matrix(self._bundle_gram())
        p = matrix(self._linear_term())
        G = matrix(-1.0 * np.eye(self.bundle_size))
        h = matrix(np.zeros(self.bundle_size))
        A = matrix(np.ones(self.bundle_size), (1, self.bundle_size))
        b = matrix(1.0)
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, G, h, A, b)
        self.alpha = np.asarray(sol['x'], dtype=np.float64)[0]

        p = np.zeros(len(self.x), dtype=np.float64)
        for ii in range(len(self.alpha)):
            p += self.alpha[ii] * (self.bundle_g1_infos_cc[ii] + self.bundle_g1_infos_ncc[ii])
        new_x = self.stability_center + 1.0/self.mu * (self.g2_l - p)
        return new_x

    ####################################################################################
    # Update Bundle Items
    ####################################################################################

    def _get_new_linearization(self, x):

        f1_cc = self.oracle.f1_cc_part(x)
        f1_ncc = self.oracle.f1_ncc_part(x)
        g1_cc = self.oracle.g1_cc_part(x)
        g1_ncc = self.oracle.g1_ncc_part(x)

        self.bundle_size += 1
        self.bundle_vectors[self.bundle_size-1] = x
        self.bundle_f1_infos_cc[self.bundle_size-1] = f1_cc
        self.bundle_f1_infos_ncc[self.bundle_size-1] = f1_ncc
        self.bundle_g1_infos_cc[self.bundle_size-1] = g1_cc
        self.bundle_g1_infos_ncc[self.bundle_size-1] = g1_ncc

        return f1_cc + f1_ncc

    def _eval_serious_condition(self, new_f, new_x):
        a = self.f_stability \
            - self.kappa * self.mu_low * 0.5 * np.linalg.norm(new_x - self.stability_center) ** 2
        return new_f <= a

    def _update_bundle_items(self, new_x):

        if self.is_serious_step:
            self._restart_bundle()
            self._update_stability_center(new_x)
        else:
            self._filter_p(new_x)
            _ = self._get_new_linearization(new_x)

    def _update_stability_center(self, x):

        self.stability_center = x
        f1 = self._get_new_linearization(x)
        f2, g2 = self.oracle.first_order_oracle_f2(x)
        self.f_stability = f1 - f2
        self.g2_l = g2

    def _update_mu(self):
        if self.is_serious_step:
            self.mu = max(self.mu_dec * self.mu, self.mu_low)
        else:
            self.mu = min(self.mu_inc * self.mu, self.mu_high)

    # TODO : Handle case when gram matrix is on point which zeros f1.
    def _bundle_gram(self):
        vec_gradients = self.bundle_g1_infos_ncc[:self.bundle_size] + self.bundle_g1_infos_cc[:self.bundle_size]
        res = np.dot(vec_gradients, vec_gradients.T)
        self.scaling_term = 1.0 / np.linalg.norm(res)
        res = self.scaling_term * res.astype(float)
        return res

    # We actually solve the dual of the quadratic problem
    def _linear_term(self):
        c = self.stability_center + 1.0 / self.mu * self.g2_l
        res = self.bundle_f1_infos_cc[:self.bundle_size] + self.bundle_f1_infos_ncc[:self.bundle_size]
        res = res + _sum_dots(self.bundle_g1_infos_cc[:self.bundle_size] + self.bundle_g1_infos_ncc[:self.bundle_size],
                              self.bundle_vectors[:self.bundle_size], c)
        res = -1.0 * res.astype('float')
        res = self.scaling_term * res
        return res

    def _filter_p(self, x):

        subapprox_seq = self.bundle_f1_infos_cc[:self.bundle_size] + self.bundle_f1_infos_ncc[:self.bundle_size]
        subapprox_seq = subapprox_seq + _sum_dots(self.bundle_g1_infos_cc[:self.bundle_size] +
                                                  self.bundle_g1_infos_ncc[:self.bundle_size],
                                                  self.bundle_vectors[:self.bundle_size], x)

        i_max = np.argmax(subapprox_seq)
        f1_minus_k_cc = self.bundle_f1_infos_cc[i_max] + np.dot(self.bundle_g1_infos_cc[i_max],
                                                                x - self.bundle_vectors[i_max])
        f1_minus_k_ncc = self.bundle_f1_infos_ncc[i_max] + np.dot(self.bundle_g1_infos_ncc[i_max],
                                                                  x - self.bundle_vectors[i_max])

        p1_cc = np.dot(self.alpha, self.bundle_g1_infos_cc[:len(self.alpha)])
        p1_ncc = np.dot(self.alpha, self.bundle_g1_infos_ncc[:len(self.alpha)])

        self.bundle_size += 1
        self.bundle_vectors[self.bundle_size-1] = x
        self.bundle_f1_infos_cc[self.bundle_size-1] = f1_minus_k_cc
        self.bundle_f1_infos_ncc[self.bundle_size-1] = f1_minus_k_ncc
        self.bundle_g1_infos_cc[self.bundle_size-1] = p1_cc
        self.bundle_g1_infos_ncc[self.bundle_size-1] = p1_ncc

    ####################################################################################
    # RESTARTING PROCEDURE
    ####################################################################################

    def _restart_if_needed(self):

        right_quantile_stability_center = self.oracle.right_eta(self.stability_center)

        if self.is_serious_step and (self.stability_center[-1] <= 0) and (right_quantile_stability_center <= 0):
            self._restart_bundle()
            self.stability_center[-1] = right_quantile_stability_center
            self._update_stability_center(self.stability_center)

        elif self.counter % self.restarting_period == 0 or self.mu >= self.mu_high \
                or self.bundle_size >= self.max_size_bundle_set - 1:
            self.mu = self.restarting_mu
            if self.bundle_size >= self.max_size_bundle_set - 1:
                self._restart_bundle()
                self._update_stability_center(self.stability_center)

            right_quantile_stability_center = self.oracle.right_eta(self.stability_center)
            cond1 = abs(self.stability_center[-1] - right_quantile_stability_center) > self.restarting_penalty_slack
            cond2 = abs(self.stability_center[-1]) < self.restarting_epsilon_eta
            if cond1 and cond2:
                self._update_penalty_term(self.restarting_factor_pen * self.oracle.pen2)
                if self.bundle_size >= self.max_size_bundle_set - 1:
                    self._restart_bundle()
                    self._update_stability_center(self.stability_center)

    def _restart_bundle(self):
        self.bundle_f1_infos_cc = np.zeros(self.max_size_bundle_set, dtype=np.float64)
        self.bundle_f1_infos_ncc = np.zeros(self.max_size_bundle_set, dtype=np.float64)
        self.bundle_g1_infos_cc = np.zeros((self.max_size_bundle_set, len(self.x)), dtype=np.float64)
        self.bundle_g1_infos_ncc = np.zeros((self.max_size_bundle_set, len(self.x)), dtype=np.float64)
        self.bundle_vectors = np.zeros((self.max_size_bundle_set, len(self.x)), dtype=np.float64)
        self.bundle_size = 0

    def _update_penalty_term(self, pen2):

        self.bundle_f1_infos_cc = pen2/self.oracle.pen2 * self.bundle_f1_infos_cc
        self.bundle_g1_infos_cc = pen2 / self.oracle.pen2 * self.bundle_g1_infos_cc
        self.oracle.pen2 = pen2
        self._update_stability_center(self.stability_center)


@njit
def _sum_dots(u, v, c):
    res = np.zeros(len(u), dtype=np.float64)
    for ii in range(len(u)):
        res[ii] = np.dot(u[ii], c - v[ii])

    return res
