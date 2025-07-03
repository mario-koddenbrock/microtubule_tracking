import os

from data_generation.optimization.eval import evaluate_results
from data_generation.optimization.optimization import run_optimization

if __name__ == "__main__":

    output_base_dir = os.path.join("data", "optimization")

    # --- Configuration for Job A ---
    cfg_path_A = os.path.join("config", "tuning_config_A.json")
    output_dir_A = os.path.join(output_base_dir, "config_A")
    os.makedirs(output_dir_A, exist_ok=True)

    # --- Configuration for Job B ---
    cfg_path_B = os.path.join("config", "tuning_config_B.json")
    output_dir_B = os.path.join(output_base_dir, "config_B")
    os.makedirs(output_dir_B, exist_ok=True)

    run_optimization(cfg_path_A)
    # run_optimization(cfg_path_B)

    evaluate_results(cfg_path_A, output_dir_A)
    # evaluate_results(cfg_path_B, output_dir_B)