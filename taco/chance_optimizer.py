"""
.. module:: chance_optimizer
   :synopsis: Module which encapsulates the optimization procedure

.. moduleauthor:: Yassine LAGUEL <first-name DOT last-name AT univ-grenoble-alpes DOT fr>
"""

from .oracle import Oracle, FastOracle
from .algorithms.bundle import BundleAlgorithm
import numpy as np
import os


class Optimizer:
    """ Base class for optimization of chance constrained problems

        For an problem instance providing a dataset and two first order oracles :math:`f` and :math:`g`, this class
        is an interface for solving the minimization problem
        .. math::
            \min_{x \in \mathbb{R}^d f(x) \text{ s.t. } \mathbb{P}[g(x,\xi) \leq 0] \geq p

    """

    def __init__(self, problem, params=None, performance_warning=False):
        """
        :type problem: Problem
        :param problem: An instance of Problem

        :type params: dict
        :param params: A dictionnary of parameters for the optimization process

        :type performance_warning: bool
        :param performance_warning: If True, prints numba performance warnings.
        """

        if not performance_warning:
            os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = str(1)

        self.params = self._treat_params(params)
        self.problem = problem
        self.oracle = self._instantiate_oracle()
        self.algorithm = self._instantiate_algorithm()
        self.solution = None

    def run(self, verbose=False):
        """
            Runs the selected algorithm to solve the chance constrained problem.

        :type verbose: bool
        :param verbose: If true, prints advance of the process in the console
        :return: solution of the problem
        """
        self.solution = self.algorithm.run(verbose=verbose)
        return self.solution

    @staticmethod
    def _treat_params(params=None):
        if params is None:
            res = {

                'nb_iterations': 100,
                'starting_point': np.array([0.0, 0.0, 0.0]),

                'bund_kappa': 0.1,
                'bund_starting_mu': 20.0,
                'bund_mu_low': 0.001,
                'bund_mu_high': 100000.0,
                'bund_delta_tol': 0.0000001,
                'bund_scaling_term': 1.0,
                'bund_restarting_period': 300,
                'bund_restarting_mu': 100,
                'bund_mu_inc': 1.25,
                'bund_mu_dec': 0.8,
                'bund_epsilon_alpha': 0.01,
                'bund_max_size_bundle_set': 30,

                'plm_mode': 'fixed_budget',
                'plm_sub_solver': None,
                'plm_fixed_budget_iter': 20,
                'plm_regularizer': 20,
                'plm_stepsize': 0.01,
                'plm_stepsize_decrease': 0.5,
                'plm_stepsize_period': 100,
                'plm_func_prec': 0.01,
                'numba': False
            }
        else:
            res = params
        return res

    def _instantiate_oracle(self):
        p = self.params['p']
        pen1 = self.params['pen1']
        pen2 = self.params['pen2']
        rho = self.params['superquantile_smoothing_param']

        if self.params['numba']:
            return FastOracle(self.problem, p, pen1, pen2, rho)
        return Oracle(self.problem, p, pen1, pen2, rho)

    def _instantiate_algorithm(self):

        return BundleAlgorithm(self.oracle, params=self.params)
