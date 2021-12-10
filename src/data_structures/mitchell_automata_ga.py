import sys
import json
from pathlib import Path
from typing import Callable, List
from numba import njit
from attr import attrs, attrib
from tqdm import tqdm
import pickle
import numpy as np


from ..resources.rules import *
from ..utils import utils
from ..utils.mutations import perform_crossover, perform_mutation


@attrs()
class MitchellAutomataGA:
    """
    The genetic algorithm used in Mitchell et. al. 1996,
    in the section "Evolving Cellular Automata with Genetic Algorithms"
    https://melaniemitchell.me/PapersContent/evca-review.pdf
    """

    generations: int = attrib()
    rule_pop_size: int = attrib()
    get_y_true: Callable = attrib()  # Defines what a 'correct' classification is
    generate_ics: Callable = (
        attrib()
    )  # Defines how to sample initial conditions at each generation
    N: int = attrib(default=149)
    max_steps: int = attrib(default=298)  # 2N as default
    r: int = attrib(default=3)
    ic_pop_size: int = attrib(default=100)
    rule_generation_gap: float = attrib(default=0.8)
    mutation_prob: float = attrib(default=1)
    num_mutations: int = attrib(default=2)

    num_elite: int = attrib(init=False)

    def __attrs_post_init__(self):
        self.num_elite = int(self.rule_pop_size * self.rule_generation_gap)

    def _init_rule_population(self) -> np.ndarray:
        """
        Randomly generate initial population, such that \lambda is distributed evenly between 0.0 and 1.0
        """
        print("Initializing rules...")
        bit_len = 2 ** ((2 * self.r) + 1)
        densities = np.random.uniform(
            size=self.rule_pop_size
        )  # Sample densities from uniform distribution
        population = utils.generate_biased_binary_arrs(bit_len, densities)
        return population

    def run(self, save_dir: str, multiprocess: bool = False, num_processes: int = 1):
        self.print_overview()
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        mean_fitnesses = []
        mean_densities = []
        best_rule = None
        best_score = 0
        best_score_generation = -1
        best_score_density = -1
        population = self._init_rule_population()
        for gen in tqdm(range(self.generations)):
            # Create new initial conditions for each generation
            ics = self.generate_ics(self.ic_pop_size, self.N)
            # Find the 'true' classification label for the given task
            y_true = self.get_y_true(ics)
            print("*******************")
            print("Y True Distribution:")
            print(np.bincount(y_true))
            print("*******************")
            if multiprocess:
                f = utils.multip_compute_fitness(
                    population, ics, self.max_steps, y_true, self.r, num_processes
                )
            else:
                f = utils.compute_fitness(
                    population, ics, self.max_steps, y_true, self.r
                )
            fitness_mean = f.mean()
            mean_fitnesses.append(fitness_mean)
            print(f"Average fitness at generation {gen}: {fitness_mean}")
            best_score_index = f.argmax()
            rule_densities = utils.calculate_densities(population)
            mean_densities.append(np.mean(rule_densities))
            if f[best_score_index] > best_score:
                print("************************************************")
                print(f"New best score: {f[best_score_index]}")
                print("************************************************")
                best_score = f[best_score_index]
                best_rule = population[best_score_index, :]
                best_score_density = rule_densities[best_score_index]
                best_score_generation = gen
            top_indices = utils.top_k(f, self.num_elite)
            elite = population[top_indices, :]
            elite_f = f[top_indices]
            print()
            print("Average density of elite:")
            print(utils.calculate_densities(elite).mean())
            print()
            population = perform_crossover(elite, elite_f, self.rule_pop_size)
            population = perform_mutation(
                population, self.num_mutations, self.mutation_prob
            )
        self.save_results(
            save_dir,
            population,
            best_score,
            best_rule,
            best_score_generation,
            best_score_density,
            mean_fitnesses,
            mean_densities,
        )

    def print_overview(self):
        print("Beginning experiment with:")
        print(f"\t generations: {self.generations}")
        print(f"\t rule_pop_size: {self.rule_pop_size}")
        print(f"\t get_y_true: {self.get_y_true.__name__}")
        print(f"\t generate_ics: {self.generate_ics.__name__}")
        print(f"\t N: {self.N}")
        print(f"\t max_steps: {self.max_steps}")
        print(f"\t repopulation_cutoff: {self.rule_generation_gap}")
        print(f"\t ic_pop_size: {self.ic_pop_size}")
        print(f"\t mutation_prob: {self.mutation_prob}")
        print(f"\t r: {self.r}")

    def save_results(
        self,
        save_dir: Path,
        rule_population: np.ndarray,
        best_score: float,
        best_rule: np.ndarray,
        best_score_generation: int,
        best_score_density: float,
        mean_fitnesses: List[float],
        mean_densities: List[float],
    ):
        print(f"Saving results to {save_dir.name}...")
        with open(save_dir / "final_rule_population.pkl", "wb") as f:
            pickle.dump(rule_population, f)
        with open(save_dir / "best_rule_data.json", "w") as f:
            json.dump(
                {
                    "best_score": best_score,
                    "best_rule": best_rule.tolist(),
                    "best_score_generation": best_score_generation,
                    "best_score_density": best_score_density,
                },
                f,
            )
        with open(save_dir / "mean_generation_data.json", "w") as f:
            json.dump(
                {"mean_fitnesses": mean_fitnesses, "mean_densities": mean_densities}, f
            )
        with open(save_dir / "params.json", "w") as f:
            json.dump(
                {
                    "generations": self.generations,
                    "pop_size": self.rule_pop_size,
                    "ic_test_size": self.ic_pop_size,
                    "r": self.r,
                    "N": self.N,
                    "max_steps": self.max_steps,
                    "mutation_prob": self.mutation_prob,
                },
                f,
            )
