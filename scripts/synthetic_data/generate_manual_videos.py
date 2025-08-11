import logging
import os

from config.synthetic_data import SyntheticDataConfig
from data_generation.video import generate_video
from utils.logger import setup_logging


def main(folder_path: str):
    """    Main function to evaluate synthetic data configurations.
    Args:
        folder_path (str): Path to the folder containing the synthetic data configurations.
    """
    num_examples_per_config = 10
    cfg_paths = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith(".json")
    ]

    print(f"Found {len(cfg_paths)} synthetic data configurations.")
    if not cfg_paths:
        print("No configuration files found in the specified folder.")
        return

    for cfg_path in cfg_paths:
        try:
            print(f"Processing configuration: {cfg_path}")

            # Load the configuration from the JSON file
            cfg = SyntheticDataConfig.from_json(cfg_path)
            cfg.num_frames = 1
            cfg.to_json(cfg_path)
            old_id = cfg.id
            for i in range(num_examples_per_config):
                cfg.id = f"{old_id}_{i}"
                generate_video(cfg, folder_path)

        except Exception as e:
            print(f"Error processing {cfg_path}: {e}")





if __name__ == "__main__":

    folder_path = "data/synthetic_manual"

    main(folder_path)

