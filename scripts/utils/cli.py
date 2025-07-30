import argparse
import sys
from pathlib import Path


def parse_gen_args():
    """Main function to parse arguments and generate videos."""
    parser = argparse.ArgumentParser(
        description="CLI for generating synthetic microtubule videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Required Arguments ---
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="Path to the base synthetic_config.json file."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the generated videos, masks, and ground truth files."
    )
    # --- Optional Batch Generation Arguments ---
    # CHANGED: The group is no longer required. This is the key change.
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--ids",
        nargs="+",
        type=int,
        help="Generate videos for a specific, space-separated list of IDs."
    )
    group.add_argument(
        "--count",
        type=int,
        help="Generate a specific number of videos sequentially."
    )
    # --- Other Optional Arguments ---
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="The starting ID for sequential generation. Used only with --count."
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save a copy of the specific config file used for each video in the output directory."
    )
    args = parser.parse_args()
    # --- 1. Argument Validation and Setup ---
    if not args.config.is_file():
        print(f"Error: Config file not found at '{args.config}'")
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir.resolve()}")
    return args



def parse_optimization_args():
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization and/or evaluation for microtubule tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show default values in help
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

    return run_optimization_flag, run_evaluation_flag, args.config_path


def get_run_ids(args, base_config):
    if args.ids:
        # Case 1: A specific list of IDs is provided.
        ids_to_generate = sorted(list(set(args.ids)))  # Sort and remove duplicates
        print(f"Mode: Generating {len(ids_to_generate)} specific videos for IDs: {ids_to_generate}")
    elif args.count:
        # Case 2: A number of sequential videos are requested.
        if args.count <= 0:
            print("Error: --count must be a positive integer.")
            sys.exit(1)
        ids_to_generate = range(args.start_id, args.start_id + args.count)
        print(f"Mode: Generating {args.count} videos sequentially, starting from ID {args.start_id}.")
    else:
        # Case 3: Neither --ids nor --count is provided.
        # Use the ID from the loaded configuration file.
        ids_to_generate = [base_config.id]
        print(f"Mode: Generating a single video using the ID from the config file: {base_config.id}")
    return ids_to_generate