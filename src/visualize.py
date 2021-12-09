import numpy as np
import sys
from typing import Callable
import matplotlib.pyplot as plt
import json
import re
import matplotlib.animation as animation

from .utils import utils
from .utils.analysis_utils import *
from .utils.generation_funcs import majority_problem_y_act, parity_problem_y_act


def run_rule(
    bit_rule: np.ndarray,
    get_y_act: Callable,
    max_steps: int,
    size: int = 149, # Size of IC
    correct_only: bool = True,
):
    """
    Runs a rule, and returns the final 1D cellular automata matrix for visualization.
    """
    r = utils.rule_size_to_r(len(bit_rule))
    while True:
        x = utils.generate_unbiased_binary_arrs(size, 1)
        correct_classification = np.squeeze(get_y_act(x)).item()
        x = np.squeeze(x)
        all_x = np.empty((max_steps, size), dtype=np.int8)
        for i in range(max_steps):
            x = np.array(
                [
                    utils.apply_bit_rule(bit_rule, x, cell_ix, r)
                    for cell_ix in range(x.shape[0])
                ]
            )
            all_x[i, :] = x
        if correct_only:
            if all(all_x[-1, :] == correct_classification):
                return all_x, correct_classification
            else:
                print("Not correct, running again..")
        else:
            return all_x, correct_classification


def display_rule_gif(X: np.ndarray, correct: int, output_path: Path = None):
    fig, ax = plt.subplots(figsize=(10, 10))
    steps_to_show, size = X.shape
    iterations_per_frame = 1
    interval = 3

    def animate(i):
        ax.clear()
        ax.set_axis_off()
        Y = np.zeros_like(X)
        upper_boundary = (i + 1) * iterations_per_frame  # window upper boundary
        lower_boundary = (
            0 if upper_boundary <= steps_to_show else upper_boundary - steps_to_show
        )  # window lower bound.
        for t in range(lower_boundary, upper_boundary):  # assign the values
            Y[t - lower_boundary, :] = X[t, :]
        img = ax.imshow(Y, cmap="gray_r", vmin=0, vmax=1)
        plt.gcf().text(0.15, 0.1, f"True label: {correct}", fontsize=18)
        return [img]

    anim = animation.FuncAnimation(
        fig, animate, frames=149, interval=interval, blit=True
    )
    if output_path:
        print(f"Saving to {output_path.name}...")
        anim.save(output_path, writer="imagemagick", fps=10)
    plt.show()


def plot_rule(X: np.ndarray, correct: int, output_file: Union[Path, str] = None):
    fig, ax = plt.subplots()
    ax.imshow(X, cmap="gray_r", vmin=0, vmax=1)
    plt.tick_params(
        labeltop=False, labelbottom=False, labelleft=False, labelright=False
    )
    print(f"Saving to {output_file}...")
    plt.title(f"True label: {correct}", y=-0.1)
    plt.savefig(output_file, dpi=400, bbox_inches="tight", pad_inches=0.5)


def save_rule_matrix(
    output_dir: str,
    get_y_act: Callable,
    num_images: int = 3,
    correct_only: bool = True,
    bit_rule: np.ndarray = None,
    rule_dir: str = None,
    max_steps: int = 256,
):
    """
    Applies the best rule to n initial configurations, and saves images to specified output directory.
    """
    if bit_rule is None and rule_dir is None:
        raise ValueError("One of bit_rule, rule_dir must be specified")
    if bit_rule is None:
        bit_rule = get_best_rule(rule_dir)
    else:
        if not isinstance(bit_rule, np.ndarray):
            print("Converting string rule to np.ndarray...")
            bit_rule = utils.bit_string_to_numpy(bit_rule)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    r = utils.rule_size_to_r(len(bit_rule))
    for i in range(num_images):
        save_path = output_dir / f"automata{i}.png"
        X, correct = run_rule(
            bit_rule,
            max_steps=max_steps,
            get_y_act=get_y_act,
            correct_only=correct_only,
        )
        plot_rule(X, correct, output_file=save_path)


def make_rule_gif(
    get_y_act: Callable,
    num_images: int = 3,
    correct_only: bool = True,
    bit_rule: np.ndarray = None,
    rule_dir: str = None,
    max_steps: int = 149,
    output_path: str = None,
):
    """
    Applies the best rule to n initial configurations, and saves images to specified output directory.
    """
    if output_path:
        output_path = Path(output_path)
    if bit_rule is None and rule_dir is None:
        raise ValueError("One of bit_rule, rule_dir must be specified")
    if bit_rule is None:
        bit_rule = get_best_rule(rule_dir)
    r = utils.rule_size_to_r(len(bit_rule))
    for i in range(num_images):
        X, correct = run_rule(
            bit_rule,
            max_steps=max_steps,
            get_y_act=get_y_act,
            correct_only=correct_only,
        )
        display_rule_gif(X, correct, output_path=output_path)


def make_line_plot(
    save_dir: str,
    fitness_data: List[List[float]],
    density_data: List[List[float]],
    labels: List = None,
    plt_title: str = None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    xs = list(range(len(fitness_data[0])))
    fig, ax = plt.subplots()
    if labels is None:
        labels = [None] * len(fitness_data)
    print(labels)
    for y, label in zip(fitness_data, labels):
        ax.plot(xs, y, label=label)
    plt.legend()
    plt.title(f"{plt_title}: Mean Fitnesses")
    ax.set_ylim([0.0, 1.0])
    fitness_output_path = save_dir / "mean_fitnesses.png"
    print(f"Saving to {fitness_output_path}...")
    plt.savefig(fitness_output_path, dpi=400, bbox_inches="tight", pad_inches=0.5)
    density_output_path = save_dir / "mean_densities.png"
    print(f"Saving to {density_output_path}...")
    fig, ax = plt.subplots()
    for y, label in zip(density_data, labels):
        ax.plot(xs, y, label=label)
    plt.legend()
    plt.title(f"{plt_title}: Mean Densities")
    ax.set_ylim([0.0, 1.0])
    plt.savefig(density_output_path, dpi=400, bbox_inches="tight", pad_inches=0.5)


def line_plots(
    rules_dir: str, save_dir: str, ignore_dirs: List[str] = None, plt_title: str = None
):
    """
    Plots line plots of mean fitneses and mean densities across generations.
    """
    fitness_data = []
    density_data = []
    labels = []
    for subd in Path(rules_dir).iterdir():
        if not subd.is_dir():
            continue
        if ignore_dirs:
            if subd.name in ignore_dirs:
                continue
        if Path(subd / "mean_generation_data.json").is_file():
            with open(subd / "mean_generation_data.json", "r") as f:
                mean_generation_data = json.load(f)
            fitness_data.append(mean_generation_data["mean_fitnesses"])
            density_data.append(mean_generation_data["mean_densities"])
            labels.append(re.search(r"\d+$", subd.name).group())
    make_line_plot(
        save_dir, fitness_data, density_data, labels=labels, plt_title=plt_title
    )


def line_plot(rule_dir: str, save_dir: str, plt_title: str = None):
    """
    Plots line plots of mean fitneses and mean densities across generations.
    """
    fitness_data = []
    density_data = []
    rule_dir = Path(rule_dir)
    if Path(rule_dir / "mean_generation_data.json").is_file():
        with open(rule_dir / "mean_generation_data.json", "r") as f:
            mean_generation_data = json.load(f)
        fitness_data.append(mean_generation_data["mean_fitnesses"])
        density_data.append(mean_generation_data["mean_densities"])
    make_line_plot(save_dir, fitness_data, density_data, plt_title=plt_title)


def plot_zone_of_chaos(
    rule_dir: str,
    get_y_act: Callable,
    output_dir: str,
    plt_title: str,
    size: int = 128,
    max_steps: int = 256,
    r: int = 3,
    num_evals: int = 100,
    ics_per_eval: int = 100,
    **kwargs,
):
    """
    Samples fitness at various densities and plots in lineplot.
    """
    bit_rule = get_best_rule(rule_dir)
    x = []
    y = []
    for density in np.linspace(0, 1, num_evals):
        ics = utils.generate_biased_binary_arrs(
            size, np.full((ics_per_eval,), fill_value=density)
        )
        y_act = get_y_act(ics)
        f = utils.multip_compute_fitness(
            bit_rule.reshape(1, -1), ics, max_steps, y_act, r, **kwargs
        )
        y.append(1 - f.item())
        x.append(density)
        print(f"Score at {density}: {f}")

    fig, ax = plt.subplots()
    plt.plot(x, y)
    plt.ylabel("Uncertainty (1 - accuracy)")
    plt.xlabel("IC Density")
    plt.title(f"{plt_title}: 'Zone of Chaos'")
    output_path = Path(output_dir) / "zone_of_chaos.png"
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, dpi=400, bbox_inches="tight", pad_inches=0.5)
