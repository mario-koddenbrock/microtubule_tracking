import numpy as np
import optuna
from sklearn.metrics.pairwise import cosine_similarity

from config.tuning import TuningConfig
from data_generation.embeddings import ImageEmbeddingExtractor


def evaluate_similarity(
        synth_cfg,
        ref_embeddings,
        embedding_extractor: ImageEmbeddingExtractor
) -> float:
    """
    Core evaluation logic: Computes mean cosine similarity for a single config.
    (This is your old 'run' function, renamed and with an updated signature).
    """
    # Generate embeddings for the current synthetic configuration.
    synthetic_embeddings = embedding_extractor.extract_from_synthetic_config(synth_cfg)

    # For each new synthetic embedding, find its best match in the reference set.
    frame_scores = [
        np.max(cosine_similarity(emb.reshape(1, -1), ref_embeddings)[0])
        for emb in synthetic_embeddings
    ]

    # Average the scores across all generated frames.
    return float(np.mean(frame_scores))


def objective(
        trial: optuna.trial.Trial,
        tuning_cfg: TuningConfig,
        ref_embs: np.ndarray,
        embedding_extractor: ImageEmbeddingExtractor
) -> float:
    """
    Optuna objective function.

    1. Creates a synthetic data configuration using the search space defined in tuning_cfg.
    2. Evaluates its similarity to the reference embeddings.
    """
    # 1. Delegate parameter suggestion to the TuningConfig object.
    #    This is much cleaner and keeps the logic in the right place.
    synth_cfg_for_trial = tuning_cfg.create_synthetic_config_from_trial(trial)

    # 2. Evaluate this new configuration against the reference embeddings.
    return evaluate_similarity(synth_cfg_for_trial, ref_embs, embedding_extractor)

