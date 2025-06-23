import argparse
import sys
from pathlib import Path

from config.synthetic_data import SyntheticDataConfig
from data_generation.video import generate_video


def main():
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

    # --- Video ID Specification (mutually exclusive options) ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ids",
        nargs="+",
        type=int,
        help="A space-separated list of specific video series IDs to generate."
    )
    group.add_argument(
        "--count",
        type=int,
        help="The number of videos to generate sequentially."
    )

    # --- Optional Arguments ---
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

    # --- Argument Validation and Logic ---
    if not args.config.is_file():
        print(f"Error: Config file not found at '{args.config}'")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir.resolve()}")

    # Determine which video IDs to generate
    if args.ids:
        ids_to_generate = sorted(list(set(args.ids)))  # Sort and remove duplicates
    else:  # --count was used
        ids_to_generate = range(args.start_id, args.start_id + args.count)

    print(f"Preparing to generate videos for IDs: {list(ids_to_generate)}\n")

    # Load the base configuration once
    try:
        base_config = SyntheticDataConfig.load(args.config)
    except Exception as e:
        print(f"Error: Failed to load or parse the config file '{args.config}'.")
        print(f"Details: {e}")
        sys.exit(1)

    # --- Generation Loop ---
    for video_id in ids_to_generate:
        print("-" * 50)
        # Create a deep copy to prevent modifications from affecting subsequent runs
        config = base_config.copy(deep=True)
        config.id = video_id

        print(f"Generating video for series ID: {config.id}")

        # Optionally save the exact config used for this video for reproducibility
        if args.save_config:
            config_save_path = args.output_dir / f"series_{config.id}_config.json"
            config.to_json(config_save_path)
            print(f"Saved specific config to: {config_save_path}")

        # Run the generation process from your original script
        try:
            video_path, gt_path_json, _ = generate_video(config, str(args.output_dir))
            print(f"Successfully generated:")
            print(f"  - Video: {Path(video_path).name}")
            print(f"  - Ground Truth: {Path(gt_path_json).name}")
        except Exception as e:
            print(f"\nERROR: An exception occurred while generating video for ID {video_id}.")
            print(f"Details: {e}\n")
            # Decide if you want to stop or continue with the next video
            # For now, we'll print the error and continue
            continue

    print("-" * 50)
    print("All tasks completed.")


if __name__ == "__main__":
    main()