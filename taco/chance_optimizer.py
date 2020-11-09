

from .oracle import Oracle, FastOracle
from .algorithms.bundle import BundleAlgorithm
import numpy as np
import os


class Optimizer:

    def __init__(self, problem, params=None, performance_warning=False):

        if not performance_warning:
            os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = str(1)

        self.params = self._treat_params(params)
        self.problem = problem
        self.oracle = self.instantiate_oracle()
        self.algorithm = self.instantiate_algorithm()
        self.solution = None

    def run(self, verbose_mode=False):
        self.solution = self.algorithm.run(verbose_mode=verbose_mode)
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

    def instantiate_oracle(self):
        p = self.params['p']
        pen1 = self.params['pen1']
        pen2 = self.params['pen2']
        rho = self.params['superquantile_smoothing_param']

        if self.params['numba']:
            return FastOracle(self.problem, p, pen1, pen2, rho)
        return Oracle(self.problem, p, pen1, pen2, rho)

    def instantiate_algorithm(self):

        return BundleAlgorithm(self.oracle, params=self.params)
