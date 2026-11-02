"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

from typing import Dict, List, Tuple
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
        attempts = 0
        # Attempt cap: a tiny map may have fewer than beam_width distinct states.
        while len(out) < self.beam_width and attempts < 100 * self.beam_width:
            attempts += 1
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
        f = self.problem.evaluate
        tr = Tracker(self.problem)
        cost: Dict[State, float] = {}  # memoised f(x): each state is evaluated once

        def c(s: State) -> float:
            if s not in cost:
                cost[s] = f(s)
            return cost[s]

        beam = self.distinct_random_states()
        for s in beam:
            tr.offer(s, c(s))
        tr.tick()

        for _ in range(self.max_iterations):
            # dict, not set: keeps insertion order, so ties rank the same on every run.
            pool = dict.fromkeys(beam)
            for s in beam:
                pool.update(dict.fromkeys(get_neighbors(s, self.problem)))
            ranked = sorted(pool, key=c)

            # Diversity: first take the best state of each *distinct path*, and only
            # then fill remaining slots with same-path/different-time variants.
            # Stops the beam collapsing onto one route at K adjacent departure times.
            new_beam, seen_paths, rest = [], set(), []
            for s in ranked:
                if s.path not in seen_paths and len(new_beam) < self.beam_width:
                    new_beam.append(s)
                    seen_paths.add(s.path)
                else:
                    rest.append(s)
            new_beam += rest[: self.beam_width - len(new_beam)]
            new_beam.sort(key=c)

            for s in new_beam:
                tr.offer(s, c(s))
            tr.tick()

            if set(new_beam) == set(beam):  # beam stopped changing -> converged
                break
            beam = new_beam

        self.history = tr.history
        return tr.result()
