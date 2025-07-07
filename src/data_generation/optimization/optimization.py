import os
import logging
from functools import partial
from typing import Optional

import numpy as np
import optuna

from data_generation.optimization.embeddings import ImageEmbeddingExtractor
from data_generation.optimization.metrics import precompute_matric_args
from data_generation.optimization.objective import objective
from config.tuning import TuningConfig
from config.synthetic_data import SyntheticDataConfig


logger = logging.getLogger(f"mt.{__name__}")


def run_optimization(tuning_config_path: str):
    """
    Runs the Optuna optimization study and saves the results.

    This function performs the expensive model setup and optimization, then saves
    the best configuration and the Optuna study database for later evaluation.
    """
    logger.info(f"\n{'=' * 80}\nStarting OPTIMIZATION for: {tuning_config_path}\n{'=' * 80}")

    tuning_cfg: Optional[TuningConfig] = None
    embedding_extractor: Optional[ImageEmbeddingExtractor] = None
    ref_embeddings: Optional[np.ndarray] = None
    precomputed_kwargs = {}

    try:
        logger.info("--- Step 1: Loading tuning configuration ---")
        tuning_cfg = TuningConfig.load(tuning_config_path)
        logger.info(f"Tuning configuration loaded from: {tuning_config_path}")

        logger.debug("Validating tuning configuration...")
        tuning_cfg.validate()
        logger.info("Tuning configuration validated successfully.")

        # Persist the loaded config (useful if defaults were applied or overrides were passed)
        try:
            tuning_cfg.to_json(tuning_config_path)
            logger.info(f"Tuning configuration persisted to: {tuning_config_path}")
        except Exception as e:
            logger.warning(f"Could not persist tuning configuration to {tuning_config_path}: {e}", exc_info=True)
            # This is a warning, as optimization can still proceed, but the saved config might not match initial expectations.

    except FileNotFoundError:
        logger.critical(f"Tuning config file not found: {tuning_config_path}. Optimization cannot proceed.")
        return
    except ValueError as e:
        logger.critical(f"Invalid tuning configuration: {e}. Optimization cannot proceed.", exc_info=True)
        return
    except Exception as e:
        logger.critical(f"An unexpected error occurred during tuning config loading/validation: {e}", exc_info=True)
        return

    # Ensure tuning_cfg is available before proceeding
    if tuning_cfg is None:
        logger.critical("Tuning configuration not available. Exiting optimization.")
        return

    try:
        logger.info("--- Step 2: Performing model setup and reference embedding extraction ---")

        logger.debug("Initializing ImageEmbeddingExtractor (this loads the transformer model).")
        embedding_extractor = ImageEmbeddingExtractor(tuning_cfg)
        logger.info("ImageEmbeddingExtractor initialized.")

        logger.debug("Extracting reference embeddings (and fitting PCA if configured).")
        ref_embeddings = embedding_extractor.extract_from_references()
        logger.info(f"Reference embedding extraction complete. Shape: {ref_embeddings.shape}")

        logger.debug("Pre-computing metric arguments based on chosen similarity metric.")
        precomputed_kwargs = precompute_matric_args(tuning_cfg, ref_embeddings)
        logger.info(f"Metric arguments pre-computed for '{tuning_cfg.similarity_metric}'.")

    except Exception as e:
        logger.critical(f"Critical error during model setup or reference embedding extraction: {e}", exc_info=True)
        return  # Cannot proceed without extractor and reference embeddings

    # Ensure essential components are available before proceeding to optimization
    if embedding_extractor is None or ref_embeddings is None:
        logger.critical(
            "Essential components (embedding extractor or reference embeddings) are missing. Exiting optimization.")
        return

    logger.info("--- Step 3: Running Optuna optimization ---")
    db_filename = f"{tuning_cfg.output_config_id}.db"
    db_filepath = os.path.join(tuning_cfg.temp_dir, db_filename)

    try:
        os.makedirs(tuning_cfg.temp_dir, exist_ok=True)
        logger.debug(f"Ensured temporary directory exists: {tuning_cfg.temp_dir}")
    except OSError as e:
        logger.critical(
            f"Failed to create temporary directory {tuning_cfg.temp_dir}. Optimization cannot proceed. Error: {e}",
            exc_info=True)
        return

    storage_uri = f"sqlite:///{db_filepath}"
    logger.info(f"Using Optuna storage URI: {storage_uri}")

    try:
        study = optuna.create_study(
            study_name=tuning_cfg.output_config_id,
            storage=storage_uri,
            direction=tuning_cfg.direction,
            load_if_exists=tuning_cfg.load_if_exists,
        )
        logger.info(
            f"Optuna study '{tuning_cfg.output_config_id}' created/loaded. Direction: '{tuning_cfg.direction}', Load if exists: {tuning_cfg.load_if_exists}.")
        logger.info(f"Starting optimization for {tuning_cfg.num_trials} trials.")
    except Exception as e:
        logger.critical(f"Failed to create or load Optuna study: {e}", exc_info=True)
        return

    # Use partial to pass the pre-computed objects to the objective function
    objective_fcn = partial(
        objective,
        tuning_cfg=tuning_cfg,
        ref_embeddings=ref_embeddings,
        embedding_extractor=embedding_extractor,
        **precomputed_kwargs,
    )

    try:
        study.optimize(objective_fcn, n_trials=tuning_cfg.num_trials)
        logger.info(f"Optimization finished after {tuning_cfg.num_trials} trials.")
        logger.info(f"Best trial found (Trial {study.best_trial.number}): Value = {study.best_trial.value:.6f}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during Optuna optimization: {e}", exc_info=True)
        # Depending on the error, you might still want to proceed to save the best result so far
        # For now, we allow it to proceed to saving if a study object exists.

    logger.info("--- Step 4: Saving best configuration ---")

    if study is None or study.best_trial is None:
        logger.critical("No successful trials completed or best trial not found. Cannot save best configuration.")
        return

    try:
        # Ensure the directory for the output config file exists
        output_config_dir = os.path.dirname(tuning_cfg.output_config_file)
        if output_config_dir:  # Only make dir if path is not just a filename in current dir
            os.makedirs(output_config_dir, exist_ok=True)
            logger.debug(f"Ensured output config directory exists: {output_config_dir}")

        best_cfg: SyntheticDataConfig = tuning_cfg.create_synthetic_config_from_trial(study.best_trial)
        best_cfg.id = tuning_cfg.output_config_id
        best_cfg.num_frames = tuning_cfg.output_config_num_frames  # Apply desired number of frames for the final saved config
        logger.debug(
            f"Generated best SyntheticDataConfig for saving (ID: {best_cfg.id}, Num Frames: {best_cfg.num_frames}).")

        logger.debug("Validating best generated SyntheticDataConfig...")
        best_cfg.validate()
        logger.info("Best SyntheticDataConfig validated successfully.")

        best_cfg.to_json(tuning_cfg.output_config_file)
        logger.info(f"Best synthetic config saved to: {tuning_cfg.output_config_file}")

    except Exception as e:
        logger.critical(f"Critical error saving the best configuration: {e}", exc_info=True)
        # This is a critical step, if saving fails, the whole optimization might be pointless.

    logger.info(f"\n{'=' * 80}\nOPTIMIZATION process completed.\n{'=' * 80}")