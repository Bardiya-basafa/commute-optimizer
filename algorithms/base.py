"""
algorithms/base.py
Abstract base class for local search solvers.
"""

from abc import ABC, abstractmethod
import math
from typing import Tuple, List, Any, Dict
from core.state import State
from core.problem import CommuteProblem
import random

EPS = 1e-9


class Tracker:

    def __init__(self, problem: CommuteProblem):
        self.problem = problem
        self.best, self.best_cost = None, math.inf
        self.history: List[float] = []
        self.fount_at = 0

    def offer(self, s: State, c: float) -> None:
        if c < self.best_cost - EPS:
            self.best, self.best_cost, self.fount_at = s, c, len(self.history)

    def tick(self) -> None:
        self.history.append(self.best_cost)

    def result(self):
        return (self.best, self.best_cost)


class LocalSearchSolver(ABC):
    def __init__(self, problem: CommuteProblem, **kwargs):
        self.problem = problem
        self.params: Dict[str, Any] = kwargs
        self.history: List[float] = []
        self.rng = random.Random()

    @abstractmethod
    def solve(self) -> Tuple[State, float]:
        """
        Executes the search procedure.

        Returns:
            Tuple[State, float]: The best State found and its evaluated objective cost.
        """
        pass
