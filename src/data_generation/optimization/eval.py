import logging
import os
from typing import Optional, List

import numpy as np
import optuna
import optuna.visualization as vis

from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation.optimization.embeddings import ImageEmbeddingExtractor
from data_generation.video import generate_video
from plotting.plotting import visualize_embeddings

logger = logging.getLogger(f"mt.{__name__}")


def evaluate_results(tuning_config_path: str, output_dir: str):
    """
    Loads the results of a completed optimization study and generates evaluation artifacts.
    """
    logger.info(f"{'=' * 80}\nStarting EVALUATION for: {tuning_config_path}\n{'=' * 80}")


    logger.info("--- Loading configurations and study results ---")
    tuning_cfg = TuningConfig.load(tuning_config_path)

    # Ensure folders exist for output and temporary files
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tuning_cfg.temp_dir, exist_ok=True)

    # Load the completed Optuna study from its database file
    study_db_path = os.path.join(tuning_cfg.temp_dir, f'{tuning_cfg.output_config_id}.db')
    full_study_db_uri = f"sqlite:///{study_db_path}"
    logger.debug(f"Attempting to load Optuna study from: {full_study_db_uri}")

    study = optuna.load_study(study_name=tuning_cfg.output_config_id, storage=full_study_db_uri)
    # trials = study.get_trials(deepcopy=False)
    # scores = [trial.value for trial in trials if trial.value is not None]
    # max_score_idx = np.argmax(scores) if scores else None
    logger.info(f"Loaded Optuna study '{tuning_cfg.output_config_id}' from: {full_study_db_uri}")
    logger.info(f"Best trial: {study.best_trial.value:.4f} (Trial {study.best_trial.number})")


    best_cfg = tuning_cfg.create_synthetic_config_from_trial(study.best_trial)
    best_cfg.num_frames = tuning_cfg.output_config_num_frames
    best_cfg.id = tuning_cfg.output_config_id
    best_cfg.generate_microtubule_mask = False

    # # Load the best synthetic config found during optimization
    # best_cfg = SyntheticDataConfig.load(tuning_cfg.output_config_file)
    # logger.info(f"Loaded best synthetic configuration from: {tuning_cfg.output_config_file}")


    # Proceed with evaluation if all critical elements loaded
    if tuning_cfg and best_cfg and study:

        eval_config(best_cfg, tuning_cfg, output_dir)

        # Optimization history plot
        plot_output_dir = os.path.join(output_dir, "plots")

        vis.plot_optimization_history(study).write_html(os.path.join(plot_output_dir, "optimization_history.html"))
        vis.plot_param_importances(study).write_html(os.path.join(plot_output_dir, "param_importances.html"))
        vis.plot_slice(study).write_html(os.path.join(plot_output_dir, "slice_plot.html"))
        logging.info("Analysis plots saved successfully.")

    else:
        logger.error("Skipping further evaluation due to previous critical errors in loading configurations or study.")

    logger.info("Evaluation complete.")


def eval_config(cfg: SyntheticDataConfig, tuning_cfg: TuningConfig, output_dir: str):
    """
    Evaluates a specific SyntheticDataConfig against reference data.
    """
    logger.info("\n--- Setting up model for evaluation ---")
    embedding_extractor: Optional[ImageEmbeddingExtractor] = None
    try:
        embedding_extractor = ImageEmbeddingExtractor(tuning_cfg)
        logger.info("ImageEmbeddingExtractor initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize ImageEmbeddingExtractor: {e}", exc_info=True)
        raise  # Critical for evaluation, re-raise

    logger.info("\n--- Generating final video and embeddings ---")
    try:
        # Pass the full num_frames from the original best_cfg, not just num_compare_frames
        logger.debug(f"Generating video frames for final evaluation with {cfg.num_frames} frames into {output_dir}.")
        frames = generate_video(cfg, output_dir)
        logger.info(f"Generated {len(frames)} frames for the best synthetic configuration.")
    except Exception as e:
        logger.error(f"Error generating video for config ID {cfg.id}: {e}", exc_info=True)
        raise  # Critical for evaluation, re-raise

    if embedding_extractor:
        try:
            logger.debug("Extracting embeddings from reference data.")
            reference_vecs = embedding_extractor.extract_from_references()
            logger.info(f"Extracted {reference_vecs.shape[0]} reference embeddings (dim: {reference_vecs.shape[1]}).")
        except Exception as e:
            logger.error(f"Error extracting reference embeddings: {e}", exc_info=True)
            raise  # Critical for evaluation, re-raise

        try:
            logger.debug(
                f"Extracting embeddings from generated synthetic frames (first {tuning_cfg.num_compare_frames} frames).")
            # Use num_compare_frames for comparison, as defined in tuning_cfg
            synthetic_vecs = embedding_extractor.extract_from_frames(frames, tuning_cfg.num_compare_frames)
            logger.info(f"Extracted {synthetic_vecs.shape[0]} synthetic embeddings (dim: {synthetic_vecs.shape[1]}).")
        except Exception as e:
            logger.error(f"Error extracting synthetic embeddings: {e}", exc_info=True)
            raise  # Critical for evaluation, re-raise
    else:
        logger.critical("Embedding extractor not initialized. Cannot compute embeddings for visualization.")
        raise RuntimeError("Embedding extractor not available.")  # Stop if we can't get embeddings

    logger.info("\n--- Creating visualizations ---")
    plot_output_dir = os.path.join(output_dir, "plots")
    try:
        os.makedirs(plot_output_dir, exist_ok=True)
        logger.debug(f"Ensured plot output directory exists: {plot_output_dir}")
    except OSError as e:
        logger.error(f"Failed to create plot output directory {plot_output_dir}. Error: {e}", exc_info=True)
        raise  # Critical, plotting cannot proceed

    if reference_vecs is not None and synthetic_vecs is not None:
        try:
            logger.debug(
                f"Calling visualize_embeddings with reference vecs shape {reference_vecs.shape} and synthetic vecs shape {synthetic_vecs.shape}.")
            visualize_embeddings(cfg, tuning_cfg, reference_vecs, synthetic_vecs, plot_output_dir)
            logger.info(f"Embedding plot saved in {plot_output_dir}")
        except Exception as e:
            logger.error(f"Error generating embedding visualization: {e}", exc_info=True)
            # This is a plot, so you might choose not to re-raise depending on strictness
            # For now, we will re-raise to indicate a full failure of the eval_config.
            raise
    else:
        logger.error("Skipping embedding visualization: Reference or synthetic embeddings are missing.")
        raise RuntimeError("Missing embeddings for visualization.")