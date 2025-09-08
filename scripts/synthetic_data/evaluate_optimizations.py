import glob
import logging
import os
import sys

import pandas as pd

from mt.config.tuning import TuningConfig
from mt.data_generation.optimization.eval import evaluate_tuning_cfg
from mt.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def evaluate_all_configs():

    logger.info("Starting evaluation for all generated configs...")
    logger.info("-" * 80)

    # Output directory for evaluation results
    output_dir = "data/SynMT/synthetic"

    # Delete output directory if it exists
    if os.path.exists(output_dir):
        os.remove(output_dir)
        logger.info(f"Removing {output_dir}")

    # Find all generated tuning config files
    config_dir = "config/optimization"
    config_pattern = os.path.join(config_dir, "tuning_config_*.json")
    config_files = sorted(glob.glob(config_pattern))

    total_configs = len(config_files)
    if total_configs == 0:
        logger.info(f"No 'tuning_config_*.json' files found in '{config_dir}'. Exiting.")
        sys.exit(0)

    logger.info(f"Found {total_configs} config files to evaluate.")
    logger.info("-" * 80)

    all_results = []
    # Loop through all config files and run the evaluation script
    for i, config_path in enumerate(config_files, 1):
        logger.info(f"{'=' * 80}")
        logger.info(f"[{i}/{total_configs}] Evaluating config: '{config_path}'")

        try:
            # replace cluster paths if necessary
            cfg = TuningConfig.from_json(config_path)
            cfg.reference_images_dir = cfg.reference_images_dir.replace("/scratch/koddenbrock/mt", "data")

            cfg.hf_cache_dir = None
            cfg.temp_dir = ".temp"
            cfg.output_config_folder = output_dir
            cfg.output_config_num_frames = 50
            cfg.output_config_num_best = 10
            cfg.output_config_num_png = 10
            cfg.to_json(config_path)

            study_name, n_trials, best_score = evaluate_tuning_cfg(config_path, output_dir)
            all_results.append({
                "study": study_name,
                "n_trials": n_trials,
                "best_score": best_score,
            })

        except Exception as e:
            logger.error(f"Failed to evaluate config {config_path}: {e}", exc_info=True)

    logger.info("All evaluations have been completed.")

    # Create and print the results table
    results_df = pd.DataFrame(all_results)
    logger.info("\n--- Evaluation Summary ---")
    logger.info(f"\n{results_df.to_string(index=False)}")

    # Save the results to a CSV file
    csv_output_path = os.path.join(output_dir, "evaluation_summary.csv")
    results_df.to_csv(csv_output_path, index=False)
    logger.info(f"\nSummary table saved to: {csv_output_path}")


def main():
    evaluate_all_configs()


if __name__ == "__main__":
    main()