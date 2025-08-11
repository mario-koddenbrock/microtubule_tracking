import argparse
import glob
import logging
import os
import subprocess
import sys

from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def evaluate_all_configs(config_dir: str):
    """
    Finds all generated tuning configuration files and runs the evaluation for each.

    Args:
        config_dir (str): The directory containing the configuration files.
        python_script (str): The path to the Python script to execute for evaluation.
    """
    logger.info("Starting evaluation for all generated configs...")
    logger.info(f"Searching for config files in: '{config_dir}'")
    logger.info("-" * 80)


    # Find all generated tuning config files
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

        try:
            # Construct the command to run the evaluation script
            command = [
                sys.executable,  # Use the same python interpreter that runs this script
                python_script,
                "--evaluate",
                "--config",
                config_path,
            ]

            # Execute the command
            # check=True will raise a CalledProcessError if the script returns a non-zero exit code
            subprocess.run(command, check=True, text=True)

            logger.info(f"Evaluation for '{config_path}' complete.")

        except FileNotFoundError:
            logger.error(f"Error: The script '{python_script}' was not found.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            logger.error(f"An error occurred while evaluating '{config_path}'.")
            logger.error(f"Return code: {e.returncode}")
            # Continue to the next file or exit, depending on desired behavior
            # sys.exit(1) # Uncomment to stop on first error
        except Exception as e:
            logger.error(f"An unexpected error occurred for '{config_path}': {e}")
            # sys.exit(1) # Uncomment to stop on first error
        finally:
            logger.info("-" * 80)

    logger.info("All evaluations have been completed.")


def main():
    config_dir = "config/optimal_configs"
    evaluate_all_configs(config_dir)


if __name__ == "__main__":
    main()