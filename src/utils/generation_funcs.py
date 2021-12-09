import numpy as np

from . import utils


def majority_problem_y_act(ics: np.array) -> np.ndarray:
    """
    Given (ic_test_size, N) array of initial conditions, returns the expected values for the density classification (majority problem) task.
    https://en.wikipedia.org/wiki/Majority_problem_(cellular_automaton)
    The state of cells in CA should relax to all 1s if count(1s) > count(0s) in IC; otherwise, relax to all 0s.
    """
    return np.array(
        [np.argmax(np.bincount(ics[i, :])).item() for i in range(ics.shape[0])]
    )


def parity_problem_y_act(ics: np.array):
    """
    Given (ic_test_size, N) array of initial conditions, returns expected values for parity problem.
    https://en.wikipedia.org/wiki/Parity_problem_(sieve_theory)
    If initial configuration has an odd number of 1s, cell state should converge to all 1s; else, it should converge to all 0s.
    """
    return np.array([np.count_nonzero(ics[i, :]) % 2 for i in range(ics.shape[0])])


def generate_ics_majority(ic_test_size: int, N: int):
    """
    Generates initial conditions with a uniform distribution of densities.
    """
    densities = np.random.uniform(
        low=0.0, high=1.0, size=ic_test_size
    )  # Sample densities from uniform distribution
    # Create initial conditions with given densities
    return utils.generate_biased_binary_arrs(N, densities)


def generate_ics_parity(ic_test_size: int, N: int):
    """
    Generates initial conditions with uniformity in even/odd numbers of 1s.
    """
    half = int(ic_test_size / 2)
    densities = []
    e, o = 0, 0
    while e + o < ic_test_size:
        n = np.random.randint(0, N)
        if n % 2 == 0:
            if e >= half:
                continue
            e += 1
        else:
            if o >= half:
                continue
            o += 1
        densities.append(n)
    return utils.generate_biased_binary_arrs_int(N, np.array(densities))
