import argparse
import sys
from pathlib import Path

import optuna

from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation.video import generate_video


def sample_hyperparameters(
    tuning_config_path: Path,
    output_dir: Path,
) -> list[Path]:
    tuning_config = TuningConfig.from_json(tuning_config_path)

    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(sampler=sampler)
    generated_config_paths = []
    num_samples = tuning_config.num_trials
    start_id = tuning_config.output_config_id
    if not isinstance(start_id, int):
        start_id = 1000

    print(f"\nGenerating {num_samples} random parameter configurations...")
    for i in range(num_samples):
        sample_id = start_id + i
        trial = study.ask()  # Ask the sampler for a new trial (parameter set)

        sampled_config:SyntheticDataConfig = tuning_config.create_synthetic_config_from_trial(trial)
        sampled_config.id = sample_id
        sampled_config.generate_mt_mask = False
        sampled_config.num_frames = tuning_config.output_config_num_frames

        config_filename = output_dir / f"series_{sample_id:04d}_config.json"
        sampled_config.save(config_filename)
        generated_config_paths.append(config_filename)

    return generated_config_paths


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample hyperparameter configurations and generate corresponding synthetic videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tuning-config",
        type=Path,
        required=True,
        help="Path to the tuning_config.json file that defines the hyperparameter search space.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Main directory to save all outputs. Subfolders for configs and videos will be created here.",
    )

    args = parser.parse_args()

    # --- 1. Validation and Setup ---
    if not args.tuning_config.is_file():
        print(f"Error: Tuning config file not found at '{args.tuning_config}'")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Outputs will be saved in: {args.output_dir.resolve()}")

    # --- 2. Sample Hyperparameters ---
    # This step creates all the necessary config_XXX.json files.
    generated_configs = sample_hyperparameters(
        tuning_config_path=args.tuning_config,
        output_dir=args.output_dir,
    )

    # --- 3. Generate Videos ---
    # This step calls the other script in a loop.
    for config_path in generated_configs:
        cfg = SyntheticDataConfig.from_json(config_path)
        generate_video(cfg, str(args.output_dir), export_gt_data=False)

    print("\n\nScript finished successfully. All videos have been generated.")


if __name__ == "__main__":
    main()
