"""
.. module:: chance_optimizer.rst
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

        :param problem: An instance of Problem
        :param ``np.float64`` p: Safety probability threshold for the problem
        :param ``np.ndarray`` starting_point: (optional) Starting point for the algorithm
        :param ``np.float64`` pen1: (optional) First Penalization parameter
        :param ``np.float64`` pen2: (optional) Second Penalization parameter
        :param ``np.float64`` factor_pen2: (optional) Incremental factor for the second penalization parameter pen2
        :param ``np.float64`` bund_mu_start: Starting value for the proximal parameter :math:`\mu` of the bundle method
        :param ``bool``  numba: If True, instantiate an Oracle with numba in ``no-python`` mode.
        :param ``bool``  performance_warning: If True, prints numba performance warnings.
        :param ``dict`` params: Dictionnary of parameters for the optimization process
    """

    def __init__(self, problem, p=0.01, starting_point=np.zeros(3, dtype=np.float64), pen1=None, pen2=None, factor_pen2=None, bund_mu_start=None,
                 performance_warnings=False, numba=None, params=None):

        self.params = self._treat_params(p, starting_point, pen1, pen2, factor_pen2, bund_mu_start,
                                         performance_warnings, numba, params)

        if not self.params['performance_warnings']:
            os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = str(1)

        self.problem = problem
        self.oracle = self._instantiate_oracle()
        self.algorithm = self._instantiate_algorithm()
        self.solution = None

    def run(self, verbose=False):
        """
            Runs the bundle method to solve the chance constrained problem.

        :param ``bool`` verbose: If true, prints advance of the process in the console
        :return: solution of the problem
        """
        self.solution = self.algorithm.run(verbose=verbose)
        return self.solution

    def _treat_params(self, p, starting_point, pen1, pen2, factor_pen2, bund_mu_start,
                 performance_warnings, numba, params):

        arguments = locals()

        del arguments['self']
        del arguments['params']

        res = self._create_params()

        if params is not None:
            for key in params:
                res[key] = params[key]

        for given_parameter in arguments:
            if arguments[given_parameter] is not None:
                res[str(given_parameter)] = arguments[given_parameter]

        return res

    @staticmethod
    def _create_params():

        params = {
            'p': 0.03368421,
            'nb_iterations': 150,
            'starting_point': np.array([0.5, 1.25, 0.0], dtype=np.float64),

            'pen1': 100.,
            'pen2': 30.,
            'pen2_factor': 1.25,

            'bund_kappa': 0.00001,
            'bund_mu_start': 50.0,  # 50 working in about 20 iterations
            'bund_mu_low': 0.001,
            'bund_mu_high': 1000.0,
            'bund_delta_tol': 0.0000001,
            'bund_scaling_term': 1.0,
            'bund_restarting_period': 300,
            'bund_mu_restart': 50.0,  # 50 working in about 20 iterations
            'bund_mu_inc': 1.5,  # 1.5
            'bund_mu_dec': 0.66,  # 0.66
            'bund_epsilon_alpha': 0.01,
            'bund_max_size_bundle_set': 30,

            'bund_restarting_penalty_slack': 0.00001,
            'bund_restarting_epsilon_eta': 0.0001,
            'superquantile_smoothing_param': 0.1,
            'numba': False,
        }

        return params

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
