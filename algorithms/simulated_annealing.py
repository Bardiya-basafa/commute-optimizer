"""
algorithms/simulated_annealing.py
Stochastic optimization governed by a temperature schedule.
"""

import math
from typing import Tuple
from algorithms.base import LocalSearchSolver, Tracker, EPS
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class SimulatedAnnealingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp: float = self.params.get("initial_temp", 100.0)
        self.cooling_rate: float = self.params.get("cooling_rate", 0.995)
        self.min_temp: float = self.params.get("min_temp", 1e-3)
        self.max_iterations: int = self.params.get("max_iterations", 2000)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Simulated Annealing optimization.

        Guidelines:
        - Start from a high temperature where worse moves are accepted with significant probability.
        - Lower temperature over iterations according to a cooling schedule.
        - Downhill moves (improvements) must always be accepted.
        - Uphill moves (worse costs) should be accepted probabilistically according to temperature.
        - Record objective values throughout the run in self.history.

        Prohibited:
        - Ignoring the temperature parameter or turning this into pure hill climbing.
        """
        f = self.problem.evaluate
        tr = Tracker(self.problem)
        T = self.initial_temp
        cur = self.problem.generate_initial_state()
        cur_cost = f(cur)
        tr.offer(cur, cur_cost)
        for _ in range(self.max_iterations):
            if T < self.min_temp:
                break

            neighbors = get_neighbors(cur, self.problem)
            if not neighbors:
                break
            nxt = self.rng.choice(neighbors)
            nxt_cost = f(nxt)

            if nxt_cost < math.inf:
                dE = nxt_cost - cur_cost
                if dE <= 0 or self.rng.random() < math.exp(-dE / T):
                    cur, cur_cost = nxt, nxt_cost
                    tr.offer(cur, cur_cost)

            T *= self.cooling_rate
            tr.tick()
        self.history = tr.history
        return tr.result()
