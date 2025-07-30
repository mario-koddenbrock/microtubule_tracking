from data_generation.optimization.eval import evaluate_results
from data_generation.optimization.optimization import run_optimization
from scripts.utils.cli import parse_optimization_args
from scripts.utils.paths import get_output_dir_from_config_path
from utils.logger import setup_logging

logger = setup_logging()


def main():
    run_optimization_flag, run_evaluation_flag, config_path = parse_optimization_args()
    output_dir = get_output_dir_from_config_path(config_path)

    run_optimization_flag = False
    if run_optimization_flag:
        print(f"\n--- Starting Optimization for: {config_path} ---")
        run_optimization(config_path)
        print(f"--- Optimization finished for: {config_path} ---")

    if run_evaluation_flag:
        print(f"\n--- Starting Evaluation for: {config_path} ---")
        print(f"Results will be stored in: {output_dir}")
        evaluate_results(config_path, output_dir)
        print(f"--- Evaluation finished for: {config_path} ---")


if __name__ == "__main__":
    main()
