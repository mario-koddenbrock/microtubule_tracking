import os
from functools import partial

import optuna

from data_generation.optimization.embeddings import ImageEmbeddingExtractor
from data_generation.optimization.objective import objective
from config.tuning import TuningConfig


def run_optimization(tuning_config_path: str):
    """
    Runs the Optuna optimization study and saves the results.

    This function performs the expensive model setup and optimization, then saves
    the best configuration and the Optuna study database for later evaluation.
    """
    print(f"\n{'=' * 80}\nStarting OPTIMIZATION for: {tuning_config_path}\n{'=' * 80}")

    # =========================================================================
    # 1. ONE-TIME SETUP
    # =========================================================================
    print("--- Step 1: Performing model setup and reference embedding extraction ---")
    tuning_cfg = TuningConfig.load(tuning_config_path)
    tuning_cfg.validate()
    tuning_cfg.to_json(tuning_config_path)  # Persist defaults

    # Initialize the extractor (loads the model)
    embedding_extractor = ImageEmbeddingExtractor(tuning_cfg)

    # Compute reference embeddings (fits PCA if configured)
    ref_embeddings = embedding_extractor.extract_from_references()
    print(f"Setup complete. Reference embeddings shape: {ref_embeddings.shape}")

    # =========================================================================
    # 2. OPTIMIZATION
    # =========================================================================
    print("\n--- Step 2: Running Optuna optimization ---")

    db_filename = f"{tuning_cfg.output_config_id}.db"
    db_filepath = os.path.join(tuning_cfg.temp_dir, db_filename)
    os.makedirs(tuning_cfg.temp_dir, exist_ok=True)
    storage_uri = f"sqlite:///{db_filepath}"

    print(f"Using Optuna storage: {storage_uri}")

    # Create the study using the robust path
    study = optuna.create_study(
        study_name=tuning_cfg.output_config_id,
        storage=storage_uri,
        direction=tuning_cfg.direction,
        load_if_exists=True  # Good for resuming. Remember to delete the .db file after changing parameters!
    )
    # Use partial to pass the pre-computed objects to the objective function
    objective_fcn = partial(
        objective,
        tuning_cfg=tuning_cfg,
        ref_embs=ref_embeddings,
        embedding_extractor=embedding_extractor,
    )

    study.optimize(objective_fcn, n_trials=tuning_cfg.num_trials)

    # =========================================================================
    # 3. SAVE BEST CONFIGURATION
    # =========================================================================
    print("\n--- Step 3: Saving best configuration ---")

    # Ensure the output directory for the config file exists
    os.makedirs(os.path.dirname(tuning_cfg.output_config_file), exist_ok=True)

    best_cfg = tuning_cfg.create_synthetic_config_from_trial(study.best_trial)

    best_cfg.id = tuning_cfg.output_config_id
    best_cfg.num_frames = tuning_cfg.output_config_num_frames
    best_cfg.validate()
    best_cfg.to_json(tuning_cfg.output_config_file)

    print(f"✓ Best config saved to: {tuning_cfg.output_config_file}")
    print("Optimization complete.")


