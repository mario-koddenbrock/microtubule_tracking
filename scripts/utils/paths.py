import os


def get_output_dir_from_config_path(config_path: str, base_output_dir: str = os.path.join("data", "SynMT", "synthetic")) -> str:
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    if config_name.startswith("tuning_"):
        config_name = config_name.replace("tuning_", "", 1) # remove "tuning_" prefix once

    # output_dir = os.path.join(base_output_dir, config_name)
    output_dir = base_output_dir # use base output directory directly
    os.makedirs(output_dir, exist_ok=True)
    print(f"Derived evaluation output directory: {output_dir}")
    return output_dir