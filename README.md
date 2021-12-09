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

