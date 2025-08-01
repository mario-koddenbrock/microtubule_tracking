import logging
import os
from typing import Dict, Any

import optuna
import optuna.visualization as vis

from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation.optimization.embeddings import ImageEmbeddingExtractor
from data_generation.video import generate_video
from plotting.plotting import visualize_embeddings
from scripts.utils.toy_data import get_toy_data

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
    logger.info(f"Loaded Optuna study '{tuning_cfg.output_config_id}' from: {full_study_db_uri}")

    trials = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)

    # Choose top-N
    top_n = 10
    top_trials = sorted_trials[:top_n]

    for i, trial in enumerate(top_trials):
        logger.info(f"Trial {i + 1}: Value = {trial.value:.4f}, Params = {trial.params}")

        current_cfg = tuning_cfg.create_synthetic_config_from_trial(trial)
        current_cfg.num_frames = tuning_cfg.output_config_num_frames
        current_cfg.id = tuning_cfg.output_config_id
        current_cfg.generate_microtubule_mask = False

        if tuning_cfg and current_cfg and study:
            eval_config(current_cfg, tuning_cfg, output_dir)
        else:
            logger.error("Skipping further evaluation due to previous critical errors in loading configurations or study.")


    try:
        # Optimization history plot
        plot_output_dir = os.path.join(output_dir, "plots")

        vis.plot_optimization_history(study).write_html(os.path.join(plot_output_dir, "optimization_history.html"))
        vis.plot_param_importances(study).write_html(os.path.join(plot_output_dir, "param_importances.html"))
        vis.plot_slice(study).write_html(os.path.join(plot_output_dir, "slice_plot.html"))
        logging.info("Analysis plots saved successfully.")

    except Exception as e:
        logger.error(f"Failed to generate analysis plots: {e}", exc_info=True)


    logger.info("Evaluation complete.")


def eval_config(cfg: SyntheticDataConfig, tuning_cfg: TuningConfig, output_dir: str):
    """
    Evaluates a specific SyntheticDataConfig against reference data.
    """
    logger.info("\n--- Setting up model for evaluation ---")
    embedding_extractor = ImageEmbeddingExtractor(tuning_cfg)

    reference_vecs = embedding_extractor.extract_from_references()
    toy_data: Dict[str, Any] = get_toy_data(embedding_extractor)
    frames = generate_video(cfg, output_dir)
    synthetic_vecs = embedding_extractor.extract_from_frames(frames, tuning_cfg.num_compare_frames)


    logger.info("\n--- Creating visualizations ---")
    plot_output_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    visualize_embeddings(
        cfg=cfg,
        tuning_cfg=tuning_cfg,
        ref_embeddings=reference_vecs,
        synthetic_embeddings=synthetic_vecs,
        toy_data=toy_data,
        output_dir=plot_output_dir
    )
    logger.info(f"Embedding plot saved in {plot_output_dir}")