"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""

from collections import deque
import math
import random
from typing import List, Tuple
from algorithms.base import LocalSearchSolver, Tracker, EPS
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)
        self.max_steps: int = self.params.get("max_steps", 500)
        self.rng = random.Random()

    def solve(self) -> Tuple[State, float]:
        """
        Executes Hill Climbing optimization.

        Guidelines:
        - Search through the neighborhood greedily for candidates that minimize total travel time.
        - Handle flat local maxima / shoulders by allowing up to `max_sideways` non-improving steps.
        - Apply up to `max_restarts` to explore different starting basins.
        - Track the best objective cost seen at each major step in self.history.

        Prohibited:
        - Looping indefinitely on flat plateaus.
        """

        tr = Tracker(self.problem)
        f = self.problem.evaluate

        for restart in range(self.max_restarts):
            cur = (
                self.problem.generate_initial_state()
                if restart == 0
                else random_state(self.problem, self.rng)
            )
            cur_cost = f(cur)

            sideways_left = self.max_sideways
            tabu = deque([cur])
            for _ in range(self.max_steps):
                cand = [n for n in get_neighbors(cur, self.problem) if n not in tabu]
                if not cand:
                    break

                costs = [f(n) for n in cand]
                best_c = min(costs)
                nxt_states = [n for n, c in zip(cand, costs) if c <= best_c + EPS]
                nxt = self.rng.choice(seq=nxt_states)
                if best_c + EPS < cur_cost:
                    cur, cur_cost = nxt, best_c
                elif abs(best_c - cur_cost) <= EPS and sideways_left > 0:
                    cur, cur_cost = nxt, best_c
                    sideways_left -= 1
                else:
                    break

                tabu.append(cur)
                tr.offer(cur, cur_cost)
                tr.tick()
        return tr.result()
