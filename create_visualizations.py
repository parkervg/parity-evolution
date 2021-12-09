from src.visualize import *
from src.resources.rules import GKL, DAS, BFO
from src.utils.generation_funcs import majority_problem_y_act, parity_problem_y_act

if __name__ == "__main__":
    """
    Majority problem phase rule
    """
    make_rule_gif(
        rule_dir="results/ga/majority/mutation_prob_runs/300gen/0.02_mutation_prob_300_gen_0",
        get_y_act=majority_problem_y_act,
        num_images=1,
        correct_only=True,
        output_path="visualizations/phase_rule.gif",
    )
    """
    Example Parity problem rule
    """
    # make_rule_gif(
    #     rule_dir="results/ga/parity/custom_ic_generate/run0",
    #     get_y_act=parity_problem_y_act,
    #     num_images=3,
    #     correct_only=False,
    # )

    # save_rule_matrix(
    #     rule_dir="results/ga/parity/majority_ic_generate/parity_run0",
    #     output_dir="visualizations/parity/majority_ic_generate/automata_runs",
    #     get_y_act=parity_problem_y_act,
    #     num_images=5,
    #     correct_only=False,
    # )


    # save_rule_matrix(
    #     rule_dir="results/majority/0.02_mutation_prob_300_gen",
    #     output_dir="visualizations/0.72_majority/automata_runs",
    #     get_y_act = majority_problem_y_act
    #     num_images=5,
    # )
    # save_rule_matrix(
    #     rule_dir="results/parity/custom_ic_generate_r4",
    #     output_dir="visualizations/parity/automata_runs",
    #     get_y_act=parity_problem_y_act,
    #     num_images=5,
    #     correct_only=False,
    # )
    #
    # save_rule_matrix(
    #     bit_rule=BFO,
    #     output_dir="visualizations/BFO",
    #     get_y_act=parity_problem_y_act,
    #     num_images=5,
    #     correct_only=False,
    # )

    # line_plot(rule_dir="results/majority/0.02_mutation_prob_0", save_dir="visualizations/0.66_majority")
    # line_plot(rule_dir="results/majority/0.02_mutation_prob_300_gen", save_dir="visualizations/0.72_majority")
    # line_plots(
    #     rules_dir="results/ga/parity/sanity_check",
    #     save_dir="visualizations/parity/sanity_check",
    #     # ignore_dirs=["0.02_mutation_prob_300_gen", "run0", "run1", "run2", "run3", "run4", "run5"],
    #     plt_title="Parity Problem",
    # )
    # Line plots for majority problem
    # line_plots(
    #     rules_dir="results/ga/parity/majority_ic_generate/",
    #     save_dir="visualizations/parity/majority_ic_generate",
    #     # ignore_dirs=["0.02_mutation_prob_300_gen", "run0", "run1", "run2", "run3", "run4", "run5"],
    #     plt_title="Parity Problem",
    # )
    #
    # line_plots(
    #     rules_dir="results/ga/majority/mutation_prob_runs/300gen",
    #     save_dir="visualizations/majority/300gen",
    #     # ignore_dirs=["0.02_mutation_prob_300_gen", "run0", "run1", "run2", "run3", "run4", "run5"],
    #     plt_title="Majority Problem",
    # )
    #
    # # # Line plots for parity problem
    # line_plots(
    #     rules_dir="results/ga/parity/custom_ic_generate",
    #     save_dir="visualizations/parity",
    #     # ignore_dirs=["0.02_mutation_prob_300_gen"],
    #     plt_title="Parity Problem",
    # )

    # plot_zone_of_chaos("results/majority/0.02_mutation_prob_0", majority_problem_y_act, output_dir="visualizations/majority", plt_title="Majority Problem")
    # plot_zone_of_chaos(
    #     "results/parity/custom_ic_generate_r4",
    #     parity_problem_y_act,
    #     output_dir="visualizations/parity",
    #     plt_title="Parity Problem",
    # )
