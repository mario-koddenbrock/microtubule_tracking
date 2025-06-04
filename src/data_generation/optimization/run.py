import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.synthetic_data import SyntheticDataConfig
from data_generation import embeddings as embd


def run(cfg: SyntheticDataConfig,
        ref_embeddings: np.ndarray | list,
        model,
        extractor) -> float:
    """
    Run the optimization objective function to evaluate the mean cosine similarity

    Returns
    -------
    float
        Mean cosine-similarity averaged (1) over reference embeddings *and*
        (2) over the selected number of frames (tune_cfg.num_compare_frames).
    """

    # Loop through the embeddings and accumulate per-frame scores
    embeddings = embd.from_cfg(cfg, model, extractor)
    frame_scores = [np.max(cosine_similarity(emb.reshape(1, -1), ref_embeddings)[0]) for emb in embeddings]

    # Average across frames
    return float(np.mean(frame_scores))