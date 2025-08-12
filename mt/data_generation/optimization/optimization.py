import logging
import os
from functools import partial

import optuna

from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from .embeddings import ImageEmbeddingExtractor
from .metrics import precompute_matric_args
from .objective import objective

logger = logging.getLogger(f"mt.{__name__}")


def run_optimization(tuning_config_path: str):
    """
    Runs the Optuna optimization study and saves the results.

    This function performs the expensive model setup and optimization, then saves
    the best configuration and the Optuna study database for later evaluation.
    """
    logger.info(f"\n{'=' * 80}\nStarting OPTIMIZATION for: {tuning_config_path}\n{'=' * 80}")

    try:
        logger.debug("--- Step 1: Loading tuning configuration ---")
        tuning_cfg = TuningConfig.load(tuning_config_path)
        tuning_cfg.validate()
        tuning_cfg.to_json(tuning_config_path)

    except Exception as e:
        logger.critical(f"An unexpected error occurred during tuning config loading/validation: {e}", exc_info=True)
        return

    # Ensure tuning_cfg is available before proceeding
    if tuning_cfg is None:
        logger.critical("Tuning configuration not available. Exiting optimization.")
        return

    try:
        logger.debug("Performing model setup and reference embedding extraction")
        embedding_extractor = ImageEmbeddingExtractor(tuning_cfg)
        ref_embeddings = embedding_extractor.extract_from_references()
        precomputed_kwargs = precompute_matric_args(tuning_cfg, ref_embeddings)

    except Exception as e:
        logger.critical(f"Critical error during model setup or reference embedding extraction: {e}", exc_info=True)
        return  # Cannot proceed without an extractor and reference embeddings

    # Ensure essential elements are available before proceeding to optimization
    if embedding_extractor is None or ref_embeddings is None:
        logger.critical("Essential components (embedding extractor or reference embeddings) are missing.")
        return

    logger.debug("--- Step 3: Running Optuna optimization ---")
    db_filename = f"{tuning_cfg.output_config_id}.db"
    db_filepath = os.path.join(tuning_cfg.temp_dir, db_filename)

    try:
        os.makedirs(tuning_cfg.temp_dir, exist_ok=True)
        logger.debug(f"Ensured temporary directory exists: {tuning_cfg.temp_dir}")
    except OSError as e:
        logger.critical(f"Failed to create temporary directory {tuning_cfg.temp_dir}. Error: {e}", exc_info=True)
        return

    storage_uri = f"sqlite:///{db_filepath}"
    logger.debug(f"Using Optuna storage URI: {storage_uri}")

    sampler = optuna.samplers.RandomSampler()

    try:
        study = optuna.create_study(
            sampler=sampler,
            study_name=tuning_cfg.output_config_id,
            storage=storage_uri,
            direction=tuning_cfg.direction,
            load_if_exists=tuning_cfg.load_if_exists,
        )
        logger.debug(
            f"Optuna study '{tuning_cfg.output_config_id}' created/loaded. "
            f"Direction: '{tuning_cfg.direction}', "
            f"Load if exists: {tuning_cfg.load_if_exists}."
        )
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

    if study is None or len(study.trials) == 0:
        logger.critical("No successful trials completed. Cannot save best configurations.")
        return

    try:
        # Ensure the output directory exists
        os.makedirs(tuning_cfg.output_config_folder, exist_ok=True)
        logger.debug(f"Ensured output config directory exists: {tuning_cfg.output_config_folder}")

        # Get basename for naming the configs
        if tuning_cfg.reference_images_dir and os.path.isdir(tuning_cfg.reference_images_dir):
            basename = os.path.basename(tuning_cfg.reference_images_dir)
        else:
            basename = os.path.splitext(os.path.basename(tuning_cfg.reference_video_path))[0]

        # Sort trials by value (according to optimization direction)
        direction_multiplier = 1 if tuning_cfg.direction == "maximize" else -1
        sorted_trials = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value * direction_multiplier,
            reverse=True
        )

        # Get top N trials
        top_n = min(tuning_cfg.output_config_num_best, len(sorted_trials))
        if top_n == 0:
            logger.warning("No valid trials with values found. Cannot save configurations.")
            return

        logger.info(f"Saving {top_n} best configurations to {tuning_cfg.output_config_folder}")

        for rank, trial in enumerate(sorted_trials[:top_n], 1):
            config_filename = f"{basename}_rank{rank}.json"
            config_path = os.path.join(tuning_cfg.output_config_folder, config_filename)

            best_cfg: SyntheticDataConfig = tuning_cfg.create_synthetic_config_from_trial(trial)
            best_cfg.id = f"{basename}_rank{rank}"
            best_cfg.num_frames = tuning_cfg.output_config_num_frames

            logger.debug(f"Validating config for rank {rank}...")
            best_cfg.validate()

            best_cfg.to_json(config_path)
            logger.debug(f"Config for rank {rank} saved to: {config_path} (value: {trial.value:.6f})")

    except Exception as e:
        logger.critical(f"Critical error saving the best configurations: {e}", exc_info=True)

    logger.info(f"\n{'=' * 80}\nOPTIMIZATION process completed.\n{'=' * 80}")