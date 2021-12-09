from numba import njit, prange
import numpy as np
from typing import Callable, Tuple, List
from functools import partial
from multiprocessing import Process, JoinableQueue
import multiprocessing as mp

DEFAULT_PROCESSES = mp.cpu_count() - 1


@njit
def apply_bit_rule(
    rule_as_bits: np.ndarray,
    x: np.ndarray,
    index: int,
    r: int,
) -> int:
    """
    Gets context, then returns next cell state given a bit rule.
    """
    context = get_context(x, index, r)
    lookup = map_binary_str(context)
    return rule_as_bits[lookup]


@njit(fastmath=True)
def get_context(x: np.ndarray, index: int, r: int) -> np.ndarray:
    """
    Gets the context with specified neighborhood size `r`.
    Here, we manually specify indices rather than use `mode="wrap"` to get the numba speed gains.
    """
    indices = [i % x.shape[0] for i in range(index - r, index + (r + 1))]
    return x.take(indices)


@njit(fastmath=True)
def map_binary_str(x: np.ndarray) -> int:
    """
    Maps 1d binary numpy array to integer.
    """
    c, v = 0, 0
    t = x.shape[0]
    for i in range(x.shape[0]):
        if x[i]:
            c += 2 ** (t - i - 1)
        v += 1
    return c


@njit(fastmath=True)
def generate_biased_binary_arrs(size: int, densities: np.ndarray) -> np.ndarray:
    """
    Generates random binary arrays with a given density.
    Used to generate both rules and initial conditions.

    Takes densities as array of floats.
    """
    binary_arrs = np.empty((densities.shape[0], size), dtype=np.int8)
    for i in range(densities.shape[0]):
        binary_arr = np.array(
            [1] * int(size * densities[i]) + [0] * (size - (int(size * densities[i])))
        )
        np.random.shuffle(binary_arr)
        binary_arrs[i, :] = binary_arr
    return binary_arrs


@njit(fastmath=True)
def generate_biased_binary_arrs_int(size: int, densities: np.ndarray) -> np.ndarray:
    """
    Takes densities as array of ints.
    """
    binary_arrs = np.empty((densities.shape[0], size), dtype=np.int8)
    for i in range(densities.shape[0]):
        binary_arr = np.array([1] * densities[i] + [0] * (size - densities[i]))
        np.random.shuffle(binary_arr)
        binary_arrs[i, :] = binary_arr
    return binary_arrs


def generate_unbiased_binary_arrs(size: int, num_arrs: int) -> np.ndarray:
    """
    Generates random binary arrays from unbiased distribution.
    """
    return np.random.choice([0, 1], size=(num_arrs, size)).astype(np.int8)


@njit(fastmath=True)
def top_k(x: np.ndarray, k: int) -> np.ndarray:
    """
    Returns indices of top-k values from 1d numpy array.

    :param x: (N,)
    """
    return x.argsort()[-k:][::-1]


@njit
def run_rule(bit_rule: np.ndarray, x: np.array, max_steps: int, r: int):
    """
    Applies a bit rule to a given IC for specified timesteps.
    Returns final configuration.
    From Mitchell et al. 1996:
        "Iterating the CA on each IC until it arrives at a fixed point or for a maximum of M = 2N time steps"
    If current cell state and previous both all equal 0 or all equal 1, we stop early. No point in continuing.
    :param bit_rule: (rule_size,)
    :param x: (N,), the initial condition to start from.
    """
    prev_x = np.full_like(x, fill_value=-1)
    for _ in range(max_steps - 1):
        x = np.array(
            [apply_bit_rule(bit_rule, x, cell_ix, r) for cell_ix in range(x.shape[0])]
        )
        if early_stop(x, prev_x):
            return x
        prev_x = np.copy(x)
    return x


@njit
def early_stop(x: np.ndarray, prev_x: np.ndarray):
    if np.all(x == prev_x):
        return True
    return False


@njit(parallel=True)
def compute_fitness(
    population: np.ndarray, ics: np.ndarray, max_steps: int, y_true: np.ndarray, r: int
) -> np.ndarray:
    """
    - All possible input cases (2 ** 149 for N = 149) is too large to feasibly compute
    - Instead, fitness is defined as fraction of correct classifications over all possible ICs
    - Different sample chosen at each generation, making fitness function stochastic
    - ICs chosen from uniform distribution over probabilities

    Some time tests:
        - My method: 20.9934
        - pycelllib: 186.629 ):
        - My method with numba: 14.18
        - Restructuring my method to use numpy arrays everywhere: 4.23
        - Separate run_rule function with numba: 3.12
        - With prange, parallel=True, and pop_size=100:
    """
    f = np.zeros((population.shape[0]), dtype=np.float64)
    for pop_ix in prange(population.shape[0]):
        classifications = np.empty(
            ics.shape[0], dtype=np.int8
        )  # Create array with shape of num_ics to hold classification predictions
        bit_rule = population[pop_ix, :]
        for ic_ix in prange(ics.shape[0]):
            x = run_rule(bit_rule, ics[ic_ix, :], max_steps, r)
            classifications[ic_ix] = decode_final_cell_state(x)
        f[pop_ix] = np.count_nonzero(classifications == y_true) / ics.shape[0]
    return f


def multip_compute_fitness(
    population: np.ndarray,
    ics: np.ndarray,
    max_steps: int,
    y_true: np.ndarray,
    r: int,
    num_processes: int = DEFAULT_PROCESSES,
    **kwargs,  # For avg_fitness argument
):
    """
    Multiprocessed version of compute_fitness().
    """
    input_queue = JoinableQueue(maxsize=10000)
    output_queue = JoinableQueue(maxsize=10000)
    fitness_func = partial(
        _multip_compute_fitness,
        ics,
        max_steps,
        y_true,
        r,
        input_queue,
        output_queue,
        **kwargs,
    )
    processes = []
    for _ in range(num_processes):
        p = Process(target=fitness_func, daemon=True)
        p.start()
        processes.append(p)
    print("Adding rules to queue...")
    for i in range(population.shape[0]):
        input_queue.put((population[i, :], i))
    data = []
    for _ in range(population.shape[0]):
        data.append(output_queue.get())
    for p in processes:
        p.terminate()
    return reconstruct_fitness_arr(data)


def reconstruct_fitness_arr(data: List[Tuple[float, int]]):
    """
    Given the output data of a multiprocessed fitness function with (fitness, index),
    reconstructs the 1D fitness array so the indices are ordered correctly.
    """
    if type(data[0][0]) == np.ndarray:  # We didn't average fitnesses from before
        f = np.empty((len(data), data[0][0].shape[0]))
    else:
        f = np.empty((len(data)))
    for (fitness, index) in data:
        f[index] = fitness
    return f


def _multip_compute_fitness(
    ics: np.ndarray,
    max_steps: int,
    y_act: np.ndarray,
    r: int,
    input_queue: JoinableQueue,
    output_queue: JoinableQueue,
    avg_fitness: bool = True,  # Whether to average fitnesses at the end. True for GA, False for coevolution.
) -> None:
    while True:
        bit_rule, index = input_queue.get()
        classifications = np.empty(
            ics.shape[0], dtype=np.int8
        )  # Create array with shape of num_ics to hold classification predictions
        for ic_ix in range(ics.shape[0]):
            x = run_rule(bit_rule, ics[ic_ix, :], max_steps, r)
            classifications[ic_ix] = decode_final_cell_state(x)
        if avg_fitness:
            fitness = np.count_nonzero(classifications == y_act) / ics.shape[0]
        else:
            fitness = (classifications == y_act).astype(np.int8)
        output_queue.put((fitness, index))
        output_queue.task_done()


@njit
def decode_final_cell_state(x: np.ndarray) -> int:
    """
    Given final cell state, returns 1, 0, or -1 depending on configuration.
    """
    if np.all(x == 1):
        return 1
    elif np.all(x == 0):
        return 0
    else:  # Cells did not fully relax to a single state
        return -1


@njit
def calculate_densities(x: np.ndarray) -> np.ndarray:
    """
    Calculates densities of bit rules in array x of shape (num_rules, bit_rule_size)
    """
    # Since numba can't handle re-assignment to new shape array
    if x.ndim == 1:
        x_ = np.expand_dims(x, axis=0)
    else:
        x_ = x
    densities = np.zeros((x_.shape[0]), dtype=np.float64)
    rule_size = x_.shape[1]
    for i in range(x_.shape[0]):
        densities[i] = np.count_nonzero(x_[i, :]) / rule_size
    return densities


def evaluate_rules(
    population: np.ndarray,
    get_y_true: Callable,
    max_steps: int,
    n_ics: int = 10000,
    size: int = 149,
    multiprocess: bool = False,
    num_processes: int = DEFAULT_PROCESSES,
):
    """
    As in Mitchell et. al. 1996, evaluates a rule over 10 ** 4 ICs with an unbiased distribution.
    These ICs will have a density ~= 0.5, making this a lower bound for accuracy on all possible ICs (2 ** 149)
    """
    if population.ndim == 1:  # If it's just one rule
        population = np.expand_dims(population, axis=0)
    r = rule_size_to_r(population.shape[1])
    ics = generate_unbiased_binary_arrs(size, n_ics)
    y_act = get_y_true(ics)
    if multiprocess:
        return multip_compute_fitness(
            population, ics, max_steps, y_act, r, num_processes
        )
    return compute_fitness(population, ics, max_steps, y_act, r)


def bit_string_to_numpy(bit_string: str):
    """
    Transforms string form of rule (e.g. as found in resources/rules.py) to numpy array.
    """
    return (np.fromstring(bit_string, "u1") - ord("0")).astype(dtype=np.int8)


def rule_size_to_r(rule_size: int):
    """
    Given a rule size, returns the neighborhood size.
    e.g.
    >>> rule_size_to_r(128)
    >>> 3

    rule_size_to_r(512)
    >>> 4
    """
    return int(((np.log(rule_size) / np.log(2)) - 1) / 2)


@njit(fastmath=True)
def get_prob_matrix_indices(densities: np.ndarray, bins: np.ndarray):
    out = np.empty((densities.shape[0],), dtype=np.int64)
    for density_index in range(densities.shape[0]):
        for bins_index in range(bins.shape[0]):
            if bins[bins_index] < densities[density_index] < bins[bins_index+1]:
                out[density_index] = bins_index
    return out
