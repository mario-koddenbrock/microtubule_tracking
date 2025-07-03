from typing import Optional

import numpy as np
import optuna

from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation.optimization.embeddings import ImageEmbeddingExtractor
from data_generation.optimization.metrics import similarity


def objective(
    trial: optuna.trial.Trial,
    tuning_cfg: TuningConfig,
    ref_embeddings: np.ndarray,
    embedding_extractor: ImageEmbeddingExtractor,
    **precomputed_kwargs,
) -> float:

    # 1. Generate synthetic data config and embeddings for this trial
    cfg: SyntheticDataConfig = tuning_cfg.create_synthetic_config_from_trial(trial)
    num_eval_frames = getattr(tuning_cfg, 'num_eval_frames', tuning_cfg.num_compare_frames)
    synthetic_embeddings = embedding_extractor.extract_from_synthetic_config(cfg, num_eval_frames)

    # 2. Evaluate this new configuration against the reference embeddings.
    return similarity(
        tuning_cfg=tuning_cfg,
        ref_embeddings=ref_embeddings,
        synthetic_embeddings=synthetic_embeddings,
        **precomputed_kwargs,  # Pass along the dict of pre-computed args
    )


