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

    # Load the base configuration once
    try:
        base_config = SyntheticDataConfig.load(args.config)
        # base_config = SyntheticDataConfig()
        print(f"Successfully loaded base config from '{args.config}'")
        base_config.save(args.config)
    except Exception as e:
        print(f"Error: Failed to load or parse the config file '{args.config}'.")
        print(f"Details: {e}")
        sys.exit(1)

    # --- 2. Determine Which Video IDs to Generate ---
    # This logic now correctly handles the new default case.
    ids_to_generate = []
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

    # --- 3. Generation Loop ---
    for video_id in ids_to_generate:
        print(f"\n{'=' * 20} Generating Video ID: {video_id} {'=' * 20}")

        # Create a deep copy of the base config for this specific video.
        # This prevents modifications from one run affecting the next.
        config_for_id = base_config.copy(deep=True)

        # Set the correct ID for this generation run.
        config_for_id.id = video_id

        # Optionally, save the exact config used for this video for reproducibility.
        if args.save_config:
            save_path = args.output_dir / f"series_{video_id}_config.json"
            config_for_id.to_json(save_path)
            print(f"Saved specific config for this video to: {save_path}")

        # Generate the video and associated files.
        try:
            generate_video(config_for_id, str(args.output_dir), export_gt_data=False)
        except Exception as e:
            print(f"Error: An unexpected error occurred while generating video ID {video_id}.")
            print(f"Details: {e}")
            # Optionally, re-raise the exception if you want to stop the whole process on error.
            raise e

    print("\nAll requested videos have been generated.")


if __name__ == "__main__":
    main()
