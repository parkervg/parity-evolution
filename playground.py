from pathlib import Path
import numpy as np
from src.utils import utils
from src.utils import analysis_utils
from src.resources.rules import BFO
from src import visualize
from src.utils.generation_funcs import parity_problem_y_act
original_bfo_rule = utils.bit_string_to_numpy(BFO)


def get_combinations(n: int):
    """
    Gets all possible combinations for a binary string of length n
    """
    combos = []
    for i in range(int("1" * n, 2) + 1):
        combos.append(bin(i)[2:].zfill(n))
    return combos


if __name__ == "__main__":
    fitnesses = utils.evaluate_rules(
                    population=original_bfo_rule,
                    get_y_true=parity_problem_y_act,
                    max_steps=256,
                    n_ics=10000,
                    multiprocess=True,
                    num_processes=4,
                )

    # # Trying to reconstruct BFO rule in bit-string notation
    # configs = {
    #     "*11100***": 1,
    #     "11100****": 1,
    #     "*00100***": 1,
    #     "00100****": 1,
    #     "**010100*": 1,
    #     "11101****": 0,
    #     "*0101*0**": 0,
    #     "**0110***": 0,
    #     "***110110": 0,
    #     "***0110**": 0,
    #     "****1101*": 0
    # }
    # bfo_rule = np.zeros(512, dtype=np.int8)
    # bfo_rule.fill(-1)
    # index = 0
    # for config, output in configs.items():
    #     n = len([i for i in config if i =="*"])
    #     val_fills = get_combinations(n)
    #     for fill in val_fills:
    #         filled_config = []
    #         c = 0
    #         for item in config:
    #             if item == "*":
    #                 filled_config.append(fill[c])
    #                 c += 1
    #                 continue
    #             filled_config.append(item)
    #         bfo_rule[int("".join(filled_config), 2)] = output
    #
    #     # Set the empty spots to random vals, either 1 or 0
    #     assert np.all(bfo_rule[bfo_rule != original_bfo_rule] == -1)
    #     bfo_rule[bfo_rule==-1] = np.random.choice([0,1], size=len(bfo_rule[bfo_rule==-1]))
    #     X, correct = visualize.run_rule(
    #         original_bfo_rule,
    #         max_steps=256,
    #         get_y_act=parity_problem_y_act,
    #         correct_only=False,
    #         size=149
    #     )
    #     visualize.plot_rule(X, correct, output_file="test.png")

    #
    # num_astericks = sum(len([i for i in config if i =="*"]) for config in configs) # So we can spread density evenly
    # # 45 astericks, so
    # ones = int(num_astericks / 2)
    # zeros = num_astericks - ones
    # for config in configs:
    #



#
# for subd in Path("results/ga/parity/custom_ic_generate").iterdir():
#     if subd.is_file():
#         continue
#     try:
#         x = analysis_utils.get_best_rule(subd)
#         # x = np.expand_dims(x, axis=0)
#         print(utils.calculate_densities(x))
#     except ValueError:
#         pass
