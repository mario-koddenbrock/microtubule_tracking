import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.tuning import TuningConfig


def similarity(
    tuning_cfg: TuningConfig,
    ref_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
) -> float:
    """
    Core evaluation logic: Computes similarity for a config using the specified metric.
    """

    # Use the metric specified in the tuning configuration
    similarity_metric = getattr(tuning_cfg, 'similarity_similarity_metric', 'cosine')

    if similarity_metric == "mahalanobis":

        ref_mean = np.mean(ref_embeddings, axis=0)
        ref_inv_cov = np.linalg.pinv(np.cov(ref_embeddings, rowvar=False))
        return compute_mahalanobis_score(synthetic_embeddings, ref_mean, ref_inv_cov)

    elif similarity_metric == "cosine":
        return compute_cosine_score(synthetic_embeddings, ref_embeddings)

    else:
        raise ValueError(f"Unknown similarity metric: '{similarity_metric}'. Must be 'cosine' or 'mahalanobis'.")




def compute_cosine_score(synthetic_embeddings: np.ndarray, ref_embeddings: np.ndarray) -> float:
    """Computes the mean of max cosine similarities. Higher is better."""
    frame_scores = [
        np.max(cosine_similarity(emb.reshape(1, -1), ref_embeddings)[0])
        for emb in synthetic_embeddings
    ]
    return float(np.mean(frame_scores))


def compute_mahalanobis_score(
    synthetic_embeddings: np.ndarray, ref_mean: np.ndarray, ref_inv_cov: np.ndarray
) -> float:
    """
    Computes the mean negated Mahalanobis distance.
    Distance is a measure of dissimilarity (lower is better). We negate it
    so that Optuna's maximizer can work correctly (higher is better).
    """
    distances = [
        np.sqrt(np.dot(np.dot((emb - ref_mean), ref_inv_cov), (emb - ref_mean).T))
        for emb in synthetic_embeddings
    ]
    # Return the negative of the mean distance, as Optuna maximizes.
    return -float(np.mean(distances))