import numpy as np
from numba import njit


def perform_crossover(
    elite: np.ndarray,
    elite_f: np.ndarray,
    pop_size: int,
    proportional_selection: bool = True,
) -> np.ndarray:
    """
    Creates next generation by single-point crossover of randomly chosen elite rules.
    """
    print("Performing crossover...")
    new_population = np.copy(elite)
    rule_size = elite[0].shape[0]
    if proportional_selection:
        print("Using fitness proportional selection...")
        p = elite_f / elite_f.sum()  # Weight proportional to fitness
    else:
        p = [[1 / elite.shape[0] * elite.shape[0]]]  # Even distribution, summing to 1
    parent_sets = [
        elite[np.random.choice(elite.shape[0], 2, replace=True, p=p), :]
        for _ in range(pop_size - elite.shape[0])
    ]
    for parents in parent_sets:
        crossover_point = np.random.choice(rule_size)
        offspring = np.hstack(
            (parents[0][0:crossover_point], parents[1][crossover_point:])
        )
        new_population = np.vstack((offspring, new_population))
    assert pop_size == new_population.shape[0]
    return new_population


@njit(fastmath=True)
def flip_bits(x: np.ndarray, indices: np.ndarray):
    """
    Given a 1D array x, flips 0s to 1s, and 1s to 0s at specified indices.

    :param x: (N,)
    :param indices: (num_indices,)
    """
    for index in indices:
        x[index] = 1 - x[index]
    return x


def perform_mutation(population: np.ndarray, n_mutations: int = 2, p: float = 1):
    """
    With pop_size=100,
        parallel=True: 305.80 seconds
        parallel=False: 293 seconds
    Probably because with pop_size only set to 100, overhead to begin parallel processes outweighs the actual computation.
    """
    print(
        f"Performing mutations in {n_mutations} spots on each rule, with {p} probability..."
    )
    rule_size = population[0].shape[0]
    for i in range(population.shape[0]):
        # indices_to_flip = np.where(np.random.choice([0,1], p=[1 - p, p], size=rule_size) == 1)
        # population[i, :] = flip_bits(population[i, :], indices_to_flip)
        if np.random.choice([0, 1], p=[1 - p, p]):
            indices_to_flip = np.random.choice(rule_size, n_mutations, replace=False)
            population[i, :] = flip_bits(population[i, :], indices_to_flip)
    return population
