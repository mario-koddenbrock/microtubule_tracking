import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.synthetic_data import SyntheticDataConfig
from data_generation.optimization.embeddings import ImageEmbeddingExtractor


def run_objective(
    cfg: SyntheticDataConfig,
    ref_embeddings: np.ndarray,
    embedding_extractor: ImageEmbeddingExtractor
) -> float:
    """
    Evaluates the mean cosine similarity for a given synthetic configuration.

    This function is designed to be called repeatedly within an optimization loop.

    Args:
        cfg: The configuration for generating synthetic data for this evaluation.
        ref_embeddings: A pre-computed 2D numpy array of reference embeddings.
                        These embeddings may already be PCA-reduced.
        embedding_extractor: An *initialized* instance of the ImageEmbeddingExtractor class.
                             This is used to generate embeddings for the synthetic data.

    Returns:
        The mean of the maximum cosine similarities between each generated frame's
        embedding and the set of reference embeddings.
    """
    # 1. Generate embeddings for the current synthetic configuration using the extractor.
    #    The extractor will automatically apply PCA if it was fitted on the references.
    synthetic_embeddings = embedding_extractor.extract_from_synthetic_config(cfg)

    # 2. For each new synthetic embedding, find its best match in the reference set.
    frame_scores = [
        np.max(cosine_similarity(emb.reshape(1, -1), ref_embeddings)[0])
        for emb in synthetic_embeddings
    ]

    # 3. Average the scores across all generated frames.
    return float(np.mean(frame_scores))