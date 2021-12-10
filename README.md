# parity-evolution

This repo contains the source code for my COSI 217 - Adaptive Systems final project. 

![Alt Text](visualizations/phase_rule.gif)

## Summary
Since I couldn't find any great one-dimensional cellular automata libraries in Python, this code was done from scratch. The generic functions for interacting with cellular automata are found in `src/utils/utils.py`. The functions are built off of [NumPy](https://numpy.org/), and optimized (to the best of my ability) with [Numba](https://numba.pydata.org/). 

Additionally, in order to optimize evaluation of fitness at each generation, I ran the fitness evaluations in parallel using a queue-based multiprocessing strategy. 

The main programs which evolve rules for the cellular automata are found in `src/data_structures`.


### `data_structures/mitchell_automata_ga.py`
This is the basic genetic algorithm described in [Evolving Cellular Automata with Genetic Algorithms](https://melaniemitchell.me/PapersContent/evca-review.pdf). 


### `data_structures/juille_automata_coevolution.py`
This is the coevolutionary algorithm from the paper [Coevolving the "Ideal Trainer"](http://www.demo.cs.brandeis.edu/papers/gp98.pdf). 

## Usage 

#### Running Experiments 
In order to run experiments using one of the learning frameworks described above, the `run_experiments.py` file is used. 

GA refers to the genetic algorithm, CE refers to coevolutionary. 

Args for the genetic algorithm (GA) and coevolutionary algorithm (CE): 
- `generations`: int defining how many generations (or iterations) to train for 
- `rule_pop_size`: int defining how many CA rules to include in the evolving population. 
- `ic_pop_size`: int defining how many initial conditions (ICs) to use at each generation to evaluate rules. In CE, this population evolves, but in the GA this population is re-sampled from a biased distribution at each generation.
- `r`: int, the neighborhood size for the CA rules
- `get_y_true`: Callable returning the expected classification (either 1 or 0), given an initial condition.
- `generate_ics`: Callable defining how to generate new ICs
- `mutation_prob`: float, defines the probability a mutation for each CA rule or IC
- `rule_generation_gap`: float, defines how to find "elite" rules to keep at each generation. E.g. with a generation gap of 0.20, the top 80% of rules will be retained into the next generation.
- `ic_generation_gap`: **CE only**, float that defines how to find "elite" ICs at each generation. Same behavior as `rule_generation_gap`
- `num_mutations`: int, number of mutations to perform if we decide to perform a mutaiton, given our `mutation_prob`. E.g. if `num_mutations=2`, then we flip 2 bits on our bit string.
