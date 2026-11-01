"""
algorithms/genetic_algorithm.py
Population-based optimization using selection, recombination, and mutation.
"""

from typing import List, Tuple

from pyparsing import Dict
from algorithms.base import LocalSearchSolver, Tracker
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover, random_state


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)

    def distinct_random_states(self) -> List[State]:
        out = [self.problem.generate_initial_state()]
        while len(out) < self.population_size:
            s = random_state(self.problem, self.rng)
            if s not in out:
                out.append(s)

        return out

    def pick_parent(
        self, pop: List[State], costs: Dict[State, float], size: int
    ) -> State:
        return min(self.rng.sample(pop, k=size), key=costs.__getitem__)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        Guidelines:
        - Initialize a population of valid candidate solutions.
        - Design a fitness metric suitable for minimizing commute travel time (lower cost -> higher fitness).
        - Use selection operators (such as tournament or roulette selection) to pick parents.
        - Pair parents to produce offspring via `crossover()`, and apply stochastic mutation.
        - Preserve the top candidates across generations if using elitism.
        - Track best population performance per generation in self.history.

        Encouraged:
        - Experimenting with population size, selection pressure, and operator rates to balance exploration vs. exploitation.
        """
        # TODO: Implement Genetic Algorithm
        f = self.problem.evaluate
        tr = Tracker(self.problem)
        pop = self.distinct_random_states()
        while len(pop) < self.population_size:
            pop.append(random_state(self.problem, self.rng))

        for _ in range(self.generations):
            costs = {s: f(s) for s in pop}
            for s, c in costs.items():
                tr.offer(s, c)
            ranked = sorted(pop, key=costs.__getitem__)
            nxt = ranked[: self.elitism]
            while len(nxt) < self.population_size:
                p1 = self.pick_parent(pop, costs, 1)
                p2 = self.pick_parent(pop, costs, 1)
                child = (
                    crossover(p1, p2, self.problem, self.rng)
                    if self.rng.random() < self.crossover_rate
                    else p1
                )

                if self.rng.random() < self.mutation_rate:
                    neighs = get_neighbors(child, self.problem)
                    child = self.rng.choice(neighs)
                nxt.append(child)
            pop = nxt
            tr.tick()
        for s in pop:
            tr.offer(s, f(s))
        return tr.result()
