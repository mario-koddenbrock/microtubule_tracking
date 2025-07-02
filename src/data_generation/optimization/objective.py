from typing import Optional

import numpy as np
import optuna
from sklearn.metrics.pairwise import cosine_similarity

from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation.optimization.embeddings import ImageEmbeddingExtractor
from data_generation.optimization.metrics import compute_mahalanobis_score, compute_cosine_score


def evaluate_similarity(
    cfg: SyntheticDataConfig,
    tuning_cfg: TuningConfig,
    embedding_extractor: ImageEmbeddingExtractor,
    ref_embeddings: np.ndarray,
    ref_mean: Optional[np.ndarray] = None,
    ref_inv_cov: Optional[np.ndarray] = None,
) -> float:
    """
    Core evaluation logic: Computes similarity for a config using the specified metric.
    """
    # Generate embeddings for the current synthetic configuration.
    synthetic_embeddings = embedding_extractor.extract_from_synthetic_config(cfg)

    # Use the metric specified in the tuning configuration
    metric = getattr(tuning_cfg, 'similarity_metric', 'cosine')

    if metric == "mahalanobis":
        if ref_mean is None or ref_inv_cov is None:
            raise ValueError(
                "ref_mean and ref_inv_cov must be pre-computed and provided for Mahalanobis distance."
            )
        return compute_mahalanobis_score(synthetic_embeddings, ref_mean, ref_inv_cov)

    elif metric == "cosine":
        return compute_cosine_score(synthetic_embeddings, ref_embeddings)

    else:
        raise ValueError(f"Unknown similarity metric: '{metric}'. Must be 'cosine' or 'mahalanobis'.")


def objective(
        trial: optuna.trial.Trial,
        tuning_cfg: TuningConfig,
        ref_embeddings: np.ndarray,
        embedding_extractor: ImageEmbeddingExtractor
) -> float:
    """
    Evaluates the similarity for a given synthetic configuration using the
    metric specified in the tuning configuration.

    This function is a wrapper around the core evaluation logic in `objective.py`.

    Args:
        cfg: The configuration for generating synthetic data for this evaluation.
        tuning_cfg: The main tuning configuration, which specifies the similarity metric.
        ref_embeddings: A pre-computed 2D numpy array of reference embeddings.
        embedding_extractor: An *initialized* instance of the ImageEmbeddingExtractor class.

    Returns:
        The similarity score, where higher is better. For Mahalanobis distance,
        this is the negated mean distance.
    """

    ref_mean: Optional[np.ndarray] = None
    ref_inv_cov: Optional[np.ndarray] = None
    if getattr(tuning_cfg, 'similarity_metric', 'cosine') == 'mahalanobis':
        ref_mean = np.mean(ref_embeddings, axis=0)
        ref_inv_cov = np.linalg.pinv(np.cov(ref_embeddings, rowvar=False))

    # 1. Delegate parameter suggestion to the TuningConfig object.
    #    This is much cleaner and keeps the logic in the right place.
    cfg: SyntheticDataConfig = tuning_cfg.create_synthetic_config_from_trial(trial)

    # 2. Evaluate this new configuration against the reference embeddings.
    return evaluate_similarity(
        cfg=cfg,
        tuning_cfg=tuning_cfg,
        embedding_extractor=embedding_extractor,
        ref_embeddings=ref_embeddings,
        ref_mean=ref_mean,
        ref_inv_cov=ref_inv_cov,
    )

