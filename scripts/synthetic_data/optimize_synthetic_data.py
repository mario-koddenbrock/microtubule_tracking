import os
import argparse

from data_generation.optimization.eval import evaluate_results
from data_generation.optimization.optimization import run_optimization # This is your run_optimization function
from utils.logger import setup_logging

logger = setup_logging()

def get_output_dir_from_config_path(config_path: str, base_output_dir: str = os.path.join("data", "optimization")) -> str:
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    if config_name.startswith("tuning_"):
        config_name = config_name.replace("tuning_", "", 1) # remove "tuning_" prefix once
    return os.path.join(base_output_dir, config_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization and/or evaluation for microtubule tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the tuning configuration JSON file (e.g., config/tuning_config_A.json)."
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run the Optuna optimization step. If neither --optimize nor --evaluate are specified, both run by default."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the evaluation step. If neither --optimize nor --evaluate are specified, both run by default."
    )

    args = parser.parse_args()

    # Determine which actions to perform based on command-line flags
    run_optimization_flag = args.optimize
    run_evaluation_flag = args.evaluate

    # If neither --optimize nor --evaluate flags were provided, default to running both
    if not run_optimization_flag and not run_evaluation_flag:
        run_optimization_flag = True
        run_evaluation_flag = True

    output_dir_for_evaluation = get_output_dir_from_config_path(args.config_path)
    os.makedirs(output_dir_for_evaluation, exist_ok=True)
    print(f"Derived evaluation output directory: {output_dir_for_evaluation}")

    if run_optimization_flag:
        print(f"\n--- Starting Optimization for: {args.config_path} ---")
        run_optimization(args.config_path)
        print(f"--- Optimization finished for: {args.config_path} ---")

    if run_evaluation_flag:
        print(f"\n--- Starting Evaluation for: {args.config_path} ---")
        print(f"Results will be stored in: {output_dir_for_evaluation}")
        evaluate_results(args.config_path, output_dir_for_evaluation)
        print(f"--- Evaluation finished for: {args.config_path} ---")

