import glob
import logging
import os
import sys

from mt.config.tuning import TuningConfig
from mt.data_generation.optimization.eval import evaluate_results
from mt.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def evaluate_all_configs():

    logger.info("Starting evaluation for all generated configs...")
    logger.info("-" * 80)

    # Output directory for evaluation results
    output_dir = "data/SynMT/synthetic"

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

    # Loop through all config files and run the evaluation script
    for i, config_path in enumerate(config_files, 1):
        logger.info(f"[{i}/{total_configs}] Evaluating config: '{config_path}'")

        # replace cluster paths if necessary
        cfg = TuningConfig.from_json(config_path)
        cfg.reference_images_dir = cfg.reference_images_dir.replace("/scratch/koddenbrock/mt", "data")
        cfg.num_frames = 50  # This is the number of generated frames from which the synthetic data is sampled.
        cfg.hf_cache_dir = None
        cfg.temp_dir = ".temp"
        cfg.output_config_folder = output_dir
        cfg.to_json(config_path)

        evaluate_results(config_path, output_dir)

    logger.info("All evaluations have been completed.")


def main():
    evaluate_all_configs()


if __name__ == "__main__":
    main()