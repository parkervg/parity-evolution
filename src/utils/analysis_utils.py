from typing import Callable, List, Union
import numpy as np
from pathlib import Path
import pickle
import multiprocessing as mp

from . import utils

DEFAULT_PROCESSES = mp.cpu_count() - 1


def evaluate_final_fitnesses(
    results_subd: Union[Path, str],
    get_y_true: Callable,
    generate_ics: Callable,
    max_steps: int = 256,
    N: int = 149,
    k: int = 10,
    ic_test_size: int = 10000,
    multiprocess: bool = False,
    num_processes: int = DEFAULT_PROCESSES,
    overwrite: bool = False,
):
    """
    Grabs top 0.2 percent of final population, when evaluated on a set of biased IC.
    Then, applies those in top 0.2 on unbiased evaluation set of 10000 ICs.
    Returns highest scoring rule on unbiased evaluation set.
    """
    if not results_subd.is_dir():
        raise ValueError(f"Not a directory: {results_subd}")
    if (Path(results_subd) / "final_rule_population.pkl").is_file():
        subds = [results_subd]
    else:
        subds = [i for i in Path(results_subd).iterdir()]
    for subd in subds:
        if not subd.is_dir():
            continue
        if Path(subd / "final_rule_population.pkl").is_file():
            if Path(subd / "final_fitness.pkl").is_file() and not overwrite:
                print(f"Fitness file already exists for {subd.name}.")
                continue
            with open(subd / "final_rule_population.pkl", "rb") as f:
                population = pickle.load(f).astype(np.int8)
            print(f"Evaluating {subd.name}...")
            if k < population.shape[0]:  # If we need to filter down population first
                print(
                    f"k < population size ({k}, {population.shape[0]}), using biased ICs to filter before running on larger unbiased test set..."
                )
                ics = generate_ics(100, N)
                y_act = get_y_true(ics)
                r = utils.rule_size_to_r(population.shape[1])
                if multiprocess:
                    f = utils.multip_compute_fitness(
                        population,
                        ics,
                        max_steps,
                        y_act,
                        r,
                        num_processes=num_processes,
                    )
                else:
                    f = utils.compute_fitness(population, ics, max_steps, y_act, r)
                elite = population[utils.top_k(f, k), :]
            else:
                print(
                    f"No filter neccessary, running entire population of {population.shape[0]} through large "
                    f"unbiased test set..."
                )
                elite = population
            fitnesses = utils.evaluate_rules(
                population=elite,
                get_y_true=get_y_true,
                max_steps=max_steps,
                n_ics=ic_test_size,
                multiprocess=multiprocess,
                num_processes=num_processes,
            )
            print(f"Best fitness score for {subd.name}: {fitnesses.max()}")
            with open(subd / "final_fitness.pkl", "wb") as f:
                pickle.dump(fitnesses, f)
            with open(subd / "final_elite.pkl", "wb") as f:
                pickle.dump(elite, f)


def get_best_scores(results_subd: Union[Path, str]):
    """
    Reads from final_fitnesses.pkl file to get best score on evaluation data.
    """
    for subd in Path(results_subd).iterdir():
        if not subd.is_dir():
            continue
        if Path(subd / "final_fitness.pkl").is_file():
            with open(subd / "final_fitness.pkl", "rb") as f:
                final_fitness = pickle.load(f)
            print(f"{subd.name}: {final_fitness.max()}")


def get_best_rule(subd: Union[Path, str]) -> np.ndarray:
    subd = Path(subd)
    if (
        Path(subd / "final_fitness.pkl").is_file()
        and Path(subd / "final_elite.pkl").is_file()
    ):
        with open(subd / "final_fitness.pkl", "rb") as f:
            final_fitness = pickle.load(f)
        with open(subd / "final_elite.pkl", "rb") as f:
            final_elite = pickle.load(f)
        return final_elite[np.argmax(final_fitness)]
    else:
        raise ValueError(f"Missing file from {subd}")
