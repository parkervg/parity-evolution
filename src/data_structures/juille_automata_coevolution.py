import numpy as np
import multiprocessing as mp
from pathlib import Path
import pickle
from attr import attrs, attrib
from typing import Callable
from tqdm import tqdm

from ..utils import utils
from ..utils.mutations import perform_crossover, perform_mutation

DEFAULT_PROCESSES = mp.cpu_count() - 1


@attrs()
class JuilleAutomataCoevolution:
    """
    "The Ideal Trainer" Coevolutionary framework described in Juille 1998.
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
    mutation_prob: float = attrib(default=1)
    num_mutations: int = attrib(default=2)
    ic_generation_gap: float = attrib(default=0.05)
    rule_generation_gap: float = attrib(default=0.8)
    n_probability_bins: int = attrib(default=20)  # Defines how to bin probabilities, updated in prob_matrix

    num_elite_rules: int = attrib(init=False)
    num_elite_ics: int = attrib(init=False)
    bins: np.ndarray = attrib(init=False)
    total_density_pairings: np.ndarray = attrib(init=False) # All rule/IC density pairings we've seen
    density_defeats: np.ndarray = attrib(init=False) # Int. array of how many times rule has beaten IC with given density
    prob_matrix: np.ndarray = attrib(init=False) # (rule_density, ic_density), density_defeats / total_density_pairings

    def __attrs_post_init__(self):
        self.num_elite_rules = int(self.rule_pop_size * self.rule_generation_gap)
        self.num_elite_ics = int(self.ic_pop_size * self.ic_generation_gap)
        if self.num_elite_rules == 0:
            raise ValueError(f"num_elite_rules={self.num_elite_rules}")
        if self.num_elite_ics == 0:
            raise ValueError(f"num_elite_ics={self.num_elite_ics}")
        self.bins = np.linspace(0.000001, 0.999999, num=self.n_probability_bins)
        self.prob_matrix = np.empty((len(self.bins), len(self.bins)))
        self.density_defeats = np.zeros((len(self.bins), len(self.bins)))
        self.total_density_pairings = np.zeros((len(self.bins), len(self.bins)))
        self.prob_matrix.fill(0.5)

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

    def run(
            self,
            save_dir: str,
            multiprocess: bool = False,
            num_processes: int = DEFAULT_PROCESSES,
            update_every: int = 2,
    ):
        self.print_overview()
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        rule_population = self._init_rule_population()
        ic_population = self.generate_ics(self.ic_pop_size, self.N)
        for gen in tqdm(range(self.generations)):
            # Find the 'true' classification label for the given task
            y_act = self.get_y_true(ic_population)
            fitness_r, fitness_ic = self.get_fitness(
                rule_population,
                ic_population,
                self.max_steps,
                y_act,
                self.r,
                multiprocess=multiprocess,
                num_processes=num_processes,
            )
            top_rule_indices = utils.top_k(fitness_r, self.num_elite_rules)
            top_ic_indices = utils.top_k(fitness_ic, self.num_elite_ics)
            elite_rules = rule_population[top_rule_indices, :]
            elite_rules_f = fitness_r[top_rule_indices]
            elite_ics = ic_population[top_ic_indices, :]
            # Generate new rule population with mutation
            rule_population = perform_crossover(
                elite_rules, elite_rules_f, self.rule_pop_size
            )
            rule_population = perform_mutation(
                rule_population, self.num_mutations, self.mutation_prob
            )
            # Generate new IC population from resampling with biased distribution
            print("Generating new ICs...")
            new_ics = self.generate_ics(self.ic_pop_size - self.num_elite_ics, self.N)
            ic_population = np.vstack([elite_ics, new_ics])
            if gen != 0 and gen % update_every == 0:
                print("Evaluating objective fitness on random sample of ICs...")
                ics = self.generate_ics(100, self.N)
                y_act = self.get_y_true(ics)
                if multiprocess:
                    f = utils.multip_compute_fitness(
                        rule_population,
                        ics,
                        self.max_steps,
                        y_act,
                        self.r,
                        num_processes=num_processes,
                    )
                else:
                    f = utils.compute_fitness(
                        rule_population, ics, self.max_steps, y_act, self.r
                    )
                print(f"Mean fitness: {f.mean()}")
            with open(save_dir / "rule_population_checkpoint.pkl", "wb") as f:
                pickle.dump(rule_population, f)
            with open(save_dir / "final_population.pkl", "wb") as f:
                pickle.dump(ic_population, f)
        self.save_results(
            save_dir,
            rule_population,
            ic_population,
        )

    def get_fitness(
            self,
            rule_population: np.ndarray,
            ic_population: np.ndarray,
            max_steps: int,
            y_true: np.ndarray,
            r: int,
            multiprocess: bool = False,
            num_processes: int = DEFAULT_PROCESSES,
    ):
        """
        Given a rule_population of size (n_rules, bit_rule_size) and ic_population of (ic_test_size, N),
            returns (n_rules, ic_test_size) array of 1s and 0s corresponding to whether the rule beats the IC.
        """
        if rule_population.ndim == 1:
            rule_population = np.expand_dims(rule_population, axis=0)
        if ic_population.ndim == 1:
            ic_population = np.expand_dims(ic_population, axis=0)

        # RULE FITNESSES
        # (n_rules, ic_test_size)
        # # 1 if rule relaxes to correct state, 0 otherwise
        covered_r_ic = utils.multip_compute_fitness(
            rule_population, ic_population, max_steps, y_true, r, avg_fitness=False
        )
        # For each ic, we get the total number of rules that beat it
        # Then, divide by one
        w_ic = 1 / np.sum(covered_r_ic, axis=0)  # (num_ics,)
        fitness_r = np.sum(np.multiply(w_ic, covered_r_ic), axis=1)  # (num_rules,)

        # IC FITNESSES
        rule_densities = utils.calculate_densities(rule_population)
        ic_densities = utils.calculate_densities(ic_population)
        # Convert these to indices for lookup in prob_matrix
        rule_prob_matrix_indices = utils.get_prob_matrix_indices(rule_densities, self.bins)
        ic_prob_matrix_indices = utils.get_prob_matrix_indices(ic_densities, self.bins)
        entropies = self.calculate_entropies(rule_prob_matrix_indices, ic_prob_matrix_indices)

        _covered_r_ic = 1 - covered_r_ic  # Complement of covered_r_ic
        w_r = 1 / np.sum(np.multiply(_covered_r_ic, entropies, out=_covered_r_ic, where=entropies!=0), axis=1)  # (num_rules,)

        fitness_ic = np.sum(np.multiply(w_r, _covered_r_ic.T), axis=1)  # (num_ics,)
        print("********")
        print("Best IC density:")
        print(ic_densities[np.argmax(fitness_ic)])
        print("********")
        # Update prob matrix, so next run has more accurate probabilities
        self.update_prob_matrix(rule_prob_matrix_indices, ic_prob_matrix_indices, covered_r_ic)
        return fitness_r, fitness_ic

    def calculate_entropies(self, rule_prob_matrix_indices: np.ndarray, ic_prob_matrix_indices: np.ndarray):
        """
        E() function, described in section 4 of Juille 1998.

        Gets probabilities by looking up self.prob_matrix
        """
        print("Calculating entropies...")
        entropies = np.zeros((rule_prob_matrix_indices.shape[0], ic_prob_matrix_indices.shape[0]))
        for rule_index in range(rule_prob_matrix_indices.shape[0]):
            for ic_index in range(ic_prob_matrix_indices.shape[0]):
                # Lookup probability of rule beating ic
                p = self.prob_matrix[rule_prob_matrix_indices[rule_index], ic_prob_matrix_indices[ic_index]]
                entropies[rule_index, ic_index] = np.log(2) + (p * np.log(p)) + ((1-p) * np.log(1-p))
        return entropies


    def update_prob_matrix(self, rule_prob_matrix_indices: np.ndarray, ic_prob_matrix_indices: np.ndarray, covered_r_ic: np.ndarray) -> None:
        """
        Updates prob_matrix, given the outcome of the latest run of evaluations.
        Used to calculate entropy in self.calculate_entropies.
        """
        for rule_index in range(covered_r_ic.shape[0]):
            for ic_index in range(covered_r_ic.shape[1]):
                prob_matrix_indices = (rule_prob_matrix_indices[rule_index], ic_prob_matrix_indices[ic_index])
                self.total_density_pairings[prob_matrix_indices] += 1
                self.density_defeats[prob_matrix_indices] += covered_r_ic[rule_index, ic_index] # Adds 1 if the rule defeated the IC
        # Update, but if denominator is 0, keep original self.prob_matrix value (i.e. 0.5)
        self.prob_matrix = np.divide(self.density_defeats, self.total_density_pairings, out=self.prob_matrix, where=self.total_density_pairings!=0)

    def print_overview(self):
        print("Beginning experiment with:")
        print(f"\t generations: {self.generations}")
        print(f"\t pop_size: {self.rule_pop_size}")
        print(f"\t get_y_true: {self.get_y_true.__name__}")
        print(f"\t generate_ics: {self.generate_ics.__name__}")
        print(f"\t N: {self.N}")
        print(f"\t max_steps: {self.max_steps}")
        print(f"\t rule_generation_gap: {self.rule_generation_gap}")
        print(f"\t ic_generation_gap: {self.ic_generation_gap}")
        print(f"\t ic_test_size: {self.ic_pop_size}")
        print(f"\t mutation_prob: {self.mutation_prob}")
        print(f"\t r: {self.r}")

    def save_results(
            self,
            save_dir: Path,
            rule_population: np.ndarray,
            ic_population: np.ndarray,
    ):
        print(f"Saving results to {save_dir.name}...")
        with open(save_dir / "final_rule_population.pkl", "wb") as f:
            pickle.dump(rule_population, f)
        with open(save_dir / "final_ic_population.pkl", "wb") as f:
            pickle.dump(ic_population, f)
        with open(save_dir / "final_prob_matrix.pkl", "wb") as f:
            pickle.dump(self.prob_matrix, f)
        with open(save_dir / "bins.pkl", "wb") as f:
            pickle.dump(self.bins, f)