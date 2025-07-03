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
        embedding_extractor: ImageEmbeddingExtractor
) -> float:
    """
    Evaluates the similarity for a given synthetic configuration using the
    similarity_metric specified in the tuning configuration.

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


    # 1. Delegate parameter suggestion to the TuningConfig object.
    #    This is much cleaner and keeps the logic in the right place.
    cfg: SyntheticDataConfig = tuning_cfg.create_synthetic_config_from_trial(trial)
    synthetic_embeddings = embedding_extractor.extract_from_synthetic_config(cfg, tuning_cfg.num_compare_frames)

    # 2. Evaluate this new configuration against the reference embeddings.
    return similarity(
        tuning_cfg=tuning_cfg,
        ref_embeddings=ref_embeddings,
        synthetic_embeddings=synthetic_embeddings,
    )

