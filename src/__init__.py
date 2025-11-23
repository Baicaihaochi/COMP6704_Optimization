"""
Berlin52 TSP Solution Methods
==============================

This package implements five methods for solving the Berlin52 TSP:
1. Nearest Neighbor + 2-opt
2. ILP with Lazy SEC (Gurobi)
3. Genetic Algorithm
4. Simulated Annealing
5. Adaptive Multi-Strategy Hybrid (AMSH) - Novel

Author: [Your Name]
Course: COMP6704 Advanced Topics in Optimization
Institution: The Hong Kong Polytechnic University
Semester: Fall 2025
"""

__version__ = '1.0.0'
__author__ = '[Your Name]'

from .tsp_data import TSPData
from .nn_2opt import NearestNeighbor2Opt
from .genetic_algorithm import GeneticAlgorithm
from .simulated_annealing import SimulatedAnnealing
from .adaptive_hybrid import AdaptiveMultiStrategyHybrid

# ILP solver requires Gurobi - import conditionally
try:
    from .ilp_solver import ILPLazySEC, ILPMTZ
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Warning: Gurobi not available. ILP methods will not work.")

__all__ = [
    'TSPData',
    'NearestNeighbor2Opt',
    'GeneticAlgorithm',
    'SimulatedAnnealing',
    'AdaptiveMultiStrategyHybrid',
]

if GUROBI_AVAILABLE:
    __all__.extend(['ILPLazySEC', 'ILPMTZ'])
