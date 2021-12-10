import sys
import pathlib as Path
import argparse

from src.utils.analysis_utils import *
from src.utils.generation_funcs import (
    parity_problem_y_act,
    majority_problem_y_act,
    generate_ics_majority,
    generate_ics_parity,
)

"""
Creates final_elite.pkl and final_fitness.pkl by evaluating on larger set of unbiased ICs.

Usage:
    >>> python evaluate_scores.py results/majority
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate rules on an unbiased set of ICs."
    )
    parser.add_argument("results_subd")
    parser.add_argument("problem_type")
    parser.add_argument(
        "k",
        type=int,
        default=10,
        help="Number of rule to evaluate on larger set of unbiased ICs",
    )
    parser.add_argument(
        "-o", action="store_true", help="Whether to overwrite existing fitness files"
    )
    args = parser.parse_args()
    results_subd = Path(args.results_subd)
    problem_type = args.problem_type
    k = args.k
    overwrite = args.o
    if problem_type == "majority":
        print("Using majority problem generation functions...")
        get_y_act = majority_problem_y_act
        generate_ics = generate_ics_majority
    elif problem_type == "parity":
        print("Using parity problem generation functions...")
        get_y_act = parity_problem_y_act
        generate_ics = generate_ics_majority
    else:
        raise ValueError
    evaluate_final_fitnesses(
        results_subd,
        get_y_true=get_y_act,
        generate_ics=generate_ics,
        multiprocess=True,
        num_processes=4,
        k=k,
        overwrite=overwrite,
    )
    get_best_scores(results_subd)
