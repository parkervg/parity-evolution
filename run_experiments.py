import time
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Callable

from src.data_structures.mitchell_automata_ga import MitchellAutomataGA
from src.data_structures.juille_automata_coevolution import JuilleAutomataCoevolution
from src.utils import utils
from src.utils.generation_funcs import (
    parity_problem_y_act,
    majority_problem_y_act,
    generate_ics_majority,
    generate_ics_parity,
)


if __name__ == "__main__":
    """
    For 1 generation, pop_size = 100, ic_test_size=100:
        With multiprocessing: 98
        Without: 227
    Supported numba functions: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html

    TODO:
        - Plot "Zone of Chaos" for best majority/parity rules
            - Sample IC densities from 0.0 to 1.0
            - Evaluate rule on like 1000 of them
            - Plot inverse of those correct (1 - (correct / total)
        - TODO: maybe experiment with evolution to higher Rs
    """
    jac = JuilleAutomataCoevolution(
        generations=200,
        rule_pop_size=300,
        ic_pop_size=300,
        r=3,
        get_y_true=parity_problem_y_act,
        generate_ics=generate_ics_majority,
        mutation_prob=0.05,
        ic_generation_gap=0.05,
        rule_generation_gap=0.8,
        num_mutations=2,
    )
    jac.run(save_dir="test")

    # start_time = time.time()
    # for i in range(5):
    #     print(f"Beginning run {i}...")
    #     mag = MitchellAutomataGA(
    #         generations=100,
    #         pop_size=100,
    #         ic_test_size=100,
    #         r=3,
    #         get_y_true=parity_problem_y_act,
    #         generate_ics=generate_ics_majority,
    #         mutation_prob=0.02,
    #         rule_generation_gap=0.8,
    #         num_mutations=2,
    #     )
    #     mag.run(
    #         save_dir=f"results/ga/parity/majority_prob_generate/run{i}",
    #         multiprocess=True,
    #         num_processes=3,
    #     )
    # print("--- %s seconds ---" % (time.time() - start_time))

