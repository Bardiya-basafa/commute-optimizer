"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

from ast import List
import random
from typing import Tuple
from algorithms.base import LocalSearchSolver, Tracker, EPS
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)

    def distinct_random_states(self) -> List[State]:
        out = [self.problem.generate_initial_state()]
        while len(out) < self.beam_width:
            s = random_state(self.problem, self.rng)
            if s not in out:
                out.append(s)

        return out

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        Guidelines:
        - Maintain a set of `beam_width` parallel configurations.
        - In each iteration, generate neighbors from all active states in the beam.
        - Pool and rank all generated successors collectively, then retain the top `beam_width` states.
        - Halt when the beam ceases to improve or reaches max_iterations.
        - Record the best cost of the population at each step in self.history.

        Encouraged:
        - Ensuring diversity across states in the beam to prevent collapse into a single redundant path.

        Prohibited:
        - Treating this as K independent, non-communicating searches.
        """
        # TODO: Implement Local Beam Search
        f = self.problem.evaluate
        tr = Tracker(self.problem)
        beam = self.distinct_random_states()
        for s in beam:
            tr.offer(s, f(s))
        for _ in range(self.max_iterations):
            pool = set(beam)
            for s in beam:
                pool.update(get_neighbors(s, self.problem))
            ranked = sorted(pool, key=f)

            new_beam, per_path = [], {}
            for s in ranked:
                if len(new_beam) == self.beam_width:
                    break
                new_beam.append(s)

            for s in new_beam:
                tr.offer(s, f(s))
            beam = new_beam
            tr.tick()

        return tr.result()
