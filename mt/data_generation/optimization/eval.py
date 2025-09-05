import logging
import os
from typing import Dict, Any, List

import numpy as np
import optuna

from .embeddings import ImageEmbeddingExtractor
from .toy_data import get_toy_data
from ..video import generate_video, generate_frames
from ...config.synthetic_data import SyntheticDataConfig
from ...config.tuning import TuningConfig
from ...plotting.plotting import visualize_embeddings

logger = logging.getLogger(f"mt.{__name__}")


def evaluate_tuning_cfg(tuning_config_path: str, output_dir: str, visualize:bool=False):
    logger.debug(f"{'=' * 80}\nStarting EVALUATION for: {tuning_config_path}\n{'=' * 80}")

    logger.debug("--- Loading configurations and study results ---")
    tuning_cfg = TuningConfig.load(tuning_config_path)

    # Ensure folders exist for output and temporary files
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tuning_cfg.temp_dir, exist_ok=True)
    plot_output_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    # Initialize the ImageEmbeddingExtractor to extract embeddings from images
    if visualize:
        embedding_extractor = ImageEmbeddingExtractor(tuning_cfg)
        reference_vecs = embedding_extractor.extract_from_references()
        toy_data: Dict[str, Any] = get_toy_data()
    else:
        embedding_extractor = None
        reference_vecs = None
        toy_data = {}

    # Load the completed Optuna study from its database file
    study_db_path = os.path.join(tuning_cfg.temp_dir, f'{tuning_cfg.output_config_id}.db')

    if not os.path.exists(study_db_path):
        logger.info(f"Study database file not found: {study_db_path}")
        return tuning_cfg.output_config_id, 0, 0.0

    full_study_db_uri = f"sqlite:///{study_db_path}"
    logger.debug(f"Attempting to load Optuna study from: {full_study_db_uri}")

    study = optuna.load_study(study_name=tuning_cfg.output_config_id, storage=full_study_db_uri)
    logger.debug(f"Loaded Optuna study '{tuning_cfg.output_config_id}' from: {full_study_db_uri}")

    trials = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)

    # Choose top-N
    top_n = tuning_cfg.output_config_num_best
    top_trials = sorted_trials[:top_n]

    for i, trial in enumerate(top_trials):
        logger.info(f"Trial {i + 1}: Value = {trial.value:.4f}, Params = {trial.params}")

        current_cfg = tuning_cfg.create_synthetic_config_from_trial(trial)
        current_cfg.num_frames = tuning_cfg.output_config_num_frames
        current_cfg.id = f"{tuning_cfg.output_config_id}_rank_{i + 1}"
        current_cfg.generate_mt_mask = True
        current_cfg.generate_seed_mask = False

        evaluate_synthetic_data_cfg(
            cfg=current_cfg,
            tuning_cfg=tuning_cfg,
            output_dir=output_dir,
            plot_output_dir=plot_output_dir,
            embedding_extractor=embedding_extractor,
            reference_vecs=reference_vecs,
            toy_data=toy_data,
            is_for_expert_validation=(i == 0),  # Only the first trial is for expert validation
        )

    try:
        # Optimization history plot
        # plot_optimization_history(study).write_html(os.path.join(plot_output_dir, f"optimization_history_{study.study_name}.html"))
        # plot_param_importances(study).write_html(os.path.join(plot_output_dir, f"param_importances_{study.study_name}.html"))
        # plot_slice(study).write_html(os.path.join(plot_output_dir, f"slice_plot_{study.study_name}.html"))
        logging.debug("Analysis plots saved successfully.")

    except Exception as e:
        logger.error(f"Failed to generate analysis plots: {e}", exc_info=True)

    logger.debug("Evaluation complete.")
    return study.study_name, len(study.trials), study.best_value


def evaluate_synthetic_data_cfg(cfg: SyntheticDataConfig, tuning_cfg: TuningConfig, output_dir: str, plot_output_dir: str,
                embedding_extractor: ImageEmbeddingExtractor, reference_vecs: np.ndarray,
                toy_data: Dict[str, Any], is_for_expert_validation:bool = True,
                visualize: bool = False,
                ) -> None:
    """
    Evaluates a specific SyntheticDataConfig against reference data.
    """

    if output_dir is None:
        frames: List[np.ndarray] = []
        frame_generator = generate_frames(cfg, cfg.num_frames,
                                return_mt_mask=cfg.generate_mt_mask,
                                return_seed_mask=cfg.generate_seed_mask)

        for frame, *_ in frame_generator:
            frames.append(frame)
    else:
        frames = generate_video(cfg, output_dir,
                                num_png_frames=tuning_cfg.output_config_num_png,
                                is_for_expert_validation=is_for_expert_validation,
                                )


    if visualize:
        synthetic_vecs = embedding_extractor.extract_from_frames(frames, tuning_cfg.num_compare_frames)

        visualize_embeddings(
            cfg=cfg,
            tuning_cfg=tuning_cfg,
            ref_embeddings=reference_vecs,
            synthetic_embeddings=synthetic_vecs,
            toy_data=toy_data,
            output_dir=plot_output_dir,
        )
        logger.debug(f"Embedding plot saved in {plot_output_dir}")