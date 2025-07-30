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
    embedding_extractor = ImageEmbeddingExtractor(tuning_cfg)

    frames = generate_video(cfg, output_dir)
    reference_vecs = embedding_extractor.extract_from_references()
    synthetic_vecs = embedding_extractor.extract_from_frames(frames, tuning_cfg.num_compare_frames)

    logger.info("\n--- Creating visualizations ---")
    plot_output_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    visualize_embeddings(cfg, tuning_cfg, reference_vecs, synthetic_vecs, plot_output_dir)
    logger.info(f"Embedding plot saved in {plot_output_dir}")
