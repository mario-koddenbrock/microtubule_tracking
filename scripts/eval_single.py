import os

import optuna

from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation.optimization.eval import eval_config

def main(cfg_path: str, tuning_config_path: str, output_dir: str):
    print(f"\n{'=' * 80}\nStarting EVALUATION for: {tuning_config_path}\n{'=' * 80}")

    tuning_cfg = TuningConfig.load(tuning_config_path)

    # Load the best synthetic config found during optimization
    best_cfg = SyntheticDataConfig.load(cfg_path)
    print(f"Loaded configuration from: {cfg_path}")

    eval_config(best_cfg, tuning_cfg, output_dir)





if __name__ == "__main__":

    output_base_dir = os.path.join("data", "optimization")

    # --- Configuration for Job A ---
    tuning_cfg_path_A = os.path.join("config", "tuning_config_A.json")
    cfg_path_A = os.path.join("config", "synthetic_config_small.json")
    output_dir_A = os.path.join(output_base_dir, "config_A")
    os.makedirs(output_dir_A, exist_ok=True)

    main(cfg_path_A, tuning_cfg_path_A, output_dir_A)

    # # --- Configuration for Job B ---
    # tuning_cfg_path_B = os.path.join("config", "tuning_config_B.json")
    # cfg_path_B = os.path.join("config", "synthetic_config_large.json")
    # output_dir_B = os.path.join(output_base_dir, "config_B")
    # os.makedirs(output_dir_B, exist_ok=True)
    #
    # main(cfg_path_B, tuning_cfg_path_B, output_dir_B)