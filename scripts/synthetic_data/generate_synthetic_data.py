import sys

from config.synthetic_data import SyntheticDataConfig
from data_generation.video import generate_video
from scripts.utils.cli import parse_gen_args, get_run_ids


def main():
    args = parse_gen_args()

    # Load the base configuration once
    try:
        base_config = SyntheticDataConfig.load(args.config)
        base_config.save(args.config)
        print(f"Loaded base config from: {args.config}")
    except Exception as e:
        print(f"Error: Failed to load or parse the config file '{args.config}'.")
        print(f"Details: {e}")
        sys.exit(1)

    ids_to_generate = get_run_ids(args, base_config)

    # --- 3. Generation Loop ---
    for video_id in ids_to_generate:
        print(f"\n{'=' * 20} Generating Video ID: {video_id} {'=' * 20}")

        current_config = base_config.copy(deep=True)
        current_config.id = video_id

        if args.save_config:
            save_path = args.output_dir / f"series_{video_id}_config.json"
            current_config.to_json(save_path)
            print(f"Saved specific config for this video to: {save_path}")

        try:
            generate_video(current_config, str(args.output_dir))
        except Exception as e:
            print(f"Error: An unexpected error occurred while generating video ID {video_id}.")
            print(f"Details: {e}")
            raise e

    print("\nAll requested videos have been generated.")



if __name__ == "__main__":
    main()
