"""Core module
.. moduleauthor:: Yassine LAGUEL
"""


from .oracle import Oracle, FastOracle
from .chance_optimizer import Optimizer
from .problems.toy_problem import ToyProblem
from .algorithms.bundle import BundleAlgorithm

__all__ = ['Optimizer', 'Oracle', 'FastOracle', 'BundleAlgorithm', 'ToyProblem']
