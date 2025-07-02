import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_score(synth_embs: np.ndarray, ref_embs: np.ndarray) -> float:
    """Computes the mean of max cosine similarities. Higher is better."""
    frame_scores = [
        np.max(cosine_similarity(emb.reshape(1, -1), ref_embs)[0])
        for emb in synth_embs
    ]
    return float(np.mean(frame_scores))


def compute_mahalanobis_score(
    synth_embs: np.ndarray, ref_mean: np.ndarray, ref_inv_cov: np.ndarray
) -> float:
    """
    Computes the mean negated Mahalanobis distance.
    Distance is a measure of dissimilarity (lower is better). We negate it
    so that Optuna's maximizer can work correctly (higher is better).
    """
    distances = [
        np.sqrt(np.dot(np.dot((emb - ref_mean), ref_inv_cov), (emb - ref_mean).T))
        for emb in synth_embs
    ]
    # Return the negative of the mean distance, as Optuna maximizes.
    return -float(np.mean(distances))