import logging
from typing import Optional, List

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import sqrtm

from config.tuning import TuningConfig

logger = logging.getLogger(f"mt.{__name__}")


def similarity(
        tuning_cfg: TuningConfig,
        ref_embeddings: np.ndarray,
        synthetic_embeddings: np.ndarray,
        # --- Optional pre-computed values for optimization ---
        ref_mean: Optional[np.ndarray] = None,
        ref_inv_cov: Optional[np.ndarray] = None,
        ref_cov: Optional[np.ndarray] = None,
        ref_hist_bins: Optional[List[np.ndarray]] = None,
        ref_hist: Optional[np.ndarray] = None,
        ref_prob: Optional[np.ndarray] = None,
        ref_kid_term1: Optional[float] = None,
) -> float:
    """
    Core evaluation logic: Computes similarity using the specified metric.
    """
    similarity_metric = getattr(tuning_cfg, 'similarity_metric', 'cosine')
    logger.info(f"--- [METRIC START] ---")
    logger.info(f"Metric: '{similarity_metric}'")
    logger.debug(
        f"Ref Embeddings Shape: {ref_embeddings.shape}, Min/Max: {ref_embeddings.min():.4f}/{ref_embeddings.max():.4f}")
    logger.debug(
        f"Synth Embeddings Shape: {synthetic_embeddings.shape}, Min/Max: {synthetic_embeddings.min():.4f}/{synthetic_embeddings.max():.4f}")

    if ref_embeddings.shape[0] == 0:
        logger.warning("Reference embeddings are empty. Cannot compute similarity. Returning negative infinity.")
        return -float('inf')
    if synthetic_embeddings.shape[0] == 0:
        logger.warning("Synthetic embeddings are empty. Cannot compute similarity. Returning negative infinity.")
        return -float('inf')
    if ref_embeddings.shape[1] != synthetic_embeddings.shape[1]:
        logger.error(
            f"Embedding dimensions mismatch! Ref: {ref_embeddings.shape[1]}, Synthetic: {synthetic_embeddings.shape[1]}. Returning negative infinity.")
        return -float('inf')

    score: float
    try:
        # On-the-fly computation for one-off calls (or if pre-computed values are not passed)
        if similarity_metric == "mahalanobis":
            if ref_mean is None:
                logger.debug("[METRIC INFO] Computing ref_mean on the fly.")
                ref_mean = np.mean(ref_embeddings, axis=0)
            if ref_inv_cov is None:
                logger.debug("[METRIC INFO] Computing ref_inv_cov on the fly.")
                # Add a small regularization for stable inverse if needed
                # cov_matrix = np.cov(ref_embeddings, rowvar=False) + np.eye(ref_embeddings.shape[1]) * 1e-6
                try:
                    ref_inv_cov = np.linalg.pinv(np.cov(ref_embeddings, rowvar=False))
                except np.linalg.LinAlgError as e:
                    logger.error(f"Singular matrix encountered when computing inverse covariance for Mahalanobis. {e}",
                                 exc_info=True)
                    return -float('inf')  # Indicate a very bad score
            score = compute_mahalanobis_score(synthetic_embeddings, ref_mean, ref_inv_cov)

        elif similarity_metric == "cosine":
            score = compute_cosine_score(synthetic_embeddings, ref_embeddings)

        elif similarity_metric == "fid":
            if ref_mean is None:
                logger.debug("[METRIC INFO] Computing ref_mean for FID on the fly.")
                ref_mean = np.mean(ref_embeddings, axis=0)
            if ref_cov is None:
                logger.debug("[METRIC INFO] Computing ref_cov for FID on the fly.")
                ref_cov = np.cov(ref_embeddings, rowvar=False)
            score = compute_frechet_distance(synthetic_embeddings, ref_mean, ref_cov)

        elif similarity_metric == "kid":
            score = compute_kernel_inception_distance(synthetic_embeddings, ref_embeddings, ref_kid_term1)

        elif similarity_metric == "ndb":
            num_bins = getattr(tuning_cfg, 'num_hist_bins', 10)
            if ref_hist_bins is None:
                logger.debug(f"[METRIC INFO] Computing histogram bins (n={num_bins}) on the fly.")
                joint_embeddings = np.vstack([ref_embeddings, synthetic_embeddings])
                _, ref_hist_bins = np.histogramdd(joint_embeddings, bins=num_bins)
            if ref_hist is None:
                logger.debug("[METRIC INFO] Computing ref_hist on the fly.")
                ref_hist, _ = np.histogramdd(ref_embeddings, bins=ref_hist_bins)
            score = compute_ndb_score(synthetic_embeddings, ref_hist_bins=ref_hist_bins, ref_hist=ref_hist)

        elif similarity_metric == "jsd":
            num_bins = getattr(tuning_cfg, 'num_hist_bins', 10)
            if ref_hist_bins is None:
                logger.debug(f"[METRIC INFO] Computing histogram bins (n={num_bins}) on the fly.")
                joint_embeddings = np.vstack([ref_embeddings, synthetic_embeddings])
                _, ref_hist_bins = np.histogramdd(joint_embeddings, bins=num_bins)
            if ref_prob is None:
                logger.debug("[METRIC INFO] Computing ref_prob on the fly.")
                hist, _ = np.histogramdd(ref_embeddings, bins=ref_hist_bins)
                # Ensure sum is not zero to avoid division by zero
                if hist.sum() == 0:
                    logger.warning("Reference histogram sum is zero for JSD. Cannot compute probability distribution.")
                    return -float('inf')
                ref_prob = (hist / hist.sum()).flatten()
                ref_prob[ref_prob == 0] = 1e-10  # Prevent log(0) in JSD
            score = compute_js_divergence_score(synthetic_embeddings, ref_hist_bins=ref_hist_bins, ref_prob=ref_prob)
        else:
            msg = f"Unknown similarity metric: '{similarity_metric}'."
            logger.error(msg)
            raise ValueError(msg)
    except Exception as e:
        logger.error(f"Error computing similarity with metric '{similarity_metric}': {e}", exc_info=True)
        return -float('inf')  # Return a very bad score if computation fails

    logger.info(f"Final Score for Trial: {score:.6f}")
    logger.info(f"--- [METRIC END] ---\n")
    return score


def compute_cosine_score(synthetic_embeddings: np.ndarray, ref_embeddings: np.ndarray) -> float:
    logger.debug("[COSINE] Computing scores...")
    try:
        # Ensure consistent dimensions for cosine_similarity if embeddings are not 2D
        # Although your PCA step should make them 2D (N_samples, N_features)
        if ref_embeddings.ndim == 1:
            ref_embeddings = ref_embeddings.reshape(1, -1)

        frame_scores = []
        for i, emb in enumerate(synthetic_embeddings):
            # Reshape for single sample, ensure it's (1, N_features)
            current_score = np.max(cosine_similarity(emb.reshape(1, -1), ref_embeddings)[0])
            frame_scores.append(current_score)
            if logger.isEnabledFor(logging.DEBUG) and i < 5:  # Log first 5 for debug
                logger.debug(f"  [Emb {i}] Cosine score: {current_score:.6f}")

        mean_score = float(np.mean(frame_scores))
        logger.debug(f"[COSINE] Individual frame scores (first 5): {np.array(frame_scores[:5])}")
        logger.debug(f"[COSINE] Mean score: {mean_score:.6f}")
        return mean_score
    except Exception as e:
        logger.error(f"Error in cosine similarity computation: {e}", exc_info=True)
        return -float('inf')  # Return a bad score


def compute_mahalanobis_score(synthetic_embeddings: np.ndarray, ref_mean: np.ndarray, ref_inv_cov: np.ndarray) -> float:
    logger.debug("[MAHALANOBIS] Computing distances...")
    distances = []
    for i, emb in enumerate(synthetic_embeddings):
        try:
            delta = emb.squeeze() - ref_mean
            d_squared = np.dot(np.dot(delta, ref_inv_cov), delta.T)
            dist = np.sqrt(np.maximum(0, d_squared))
            distances.append(dist)
        except Exception as e:
            logger.error(f"Error computing Mahalanobis distance for embedding {i}: {e}", exc_info=True)
            distances.append(float('nan'))  # Append NaN to indicate failure for this embedding

    if not distances or all(np.isnan(distances)):
        logger.error("All Mahalanobis distances are NaN. Returning negative infinity.")
        return -float('inf')

    mean_distance = np.nanmean(distances)  # Use nanmean to ignore failed calculations
    final_score = -float(mean_distance)  # Mahalanobis is a distance, so we negate for maximization
    logger.debug(f"[MAHALANOBIS] Mean distance (excluding NaNs): {mean_distance:.6f}")
    logger.debug(f"[MAHALANOBIS] Final negated score: {final_score:.6f}")
    return final_score


def compute_frechet_distance(synthetic_embeddings: np.ndarray, ref_mean: np.ndarray, ref_cov: np.ndarray) -> float:
    """Computes the Fréchet distance between two sets of embeddings."""
    logger.debug("[FID] Computing Fréchet Distance...")
    try:
        mu1, sigma1 = ref_mean, ref_cov
        mu2 = np.mean(synthetic_embeddings, axis=0)
        sigma2 = np.cov(synthetic_embeddings, rowvar=False)

        # Calculate squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)

        # Calculate sqrt of product of cov matrices
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

        # Handle potential complex numbers in output
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            logger.debug("[FID] Complex number detected in sqrtm of covariance product. Taking real part.")

        # Calculate FID score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

        # Negate because Optuna maximizes, and lower FID is better
        final_score = -float(fid)

        logger.debug(f"[FID] Calculated FID: {fid:.6f}")
        logger.debug(f"[FID] Final negated score: {final_score:.6f}")
        return final_score
    except Exception as e:
        logger.error(f"Error in FID computation: {e}", exc_info=True)
        return -float('inf')


def _polynomial_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Computes a polynomial kernel between two sets of embeddings."""
    gamma = 1.0 / X.shape[1]
    return (gamma * (X @ Y.T) + 1.0) ** 3


def compute_kernel_inception_distance(synthetic_embeddings: np.ndarray, ref_embeddings: np.ndarray,
                                      ref_kid_term1: Optional[float] = None) -> float:
    """Computes the Kernel Inception Distance between two sets of embeddings."""
    logger.debug("[KID] Computing Kernel Inception Distance...")
    try:
        m, n = ref_embeddings.shape[0], synthetic_embeddings.shape[0]

        if m < 2 or n < 2:
            logger.warning(
                f"[KID] Need at least 2 samples for both reference ({m}) and synthetic ({n}) sets. Returning -inf.")
            return -float('inf')

        if ref_kid_term1 is None:
            logger.debug("[KID] Computing ref_kid_term1 on the fly.")
            k_xx = _polynomial_kernel(ref_embeddings, ref_embeddings)
            # Sum over off-diagonal elements
            term1 = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
        else:
            term1 = ref_kid_term1

        k_yy = _polynomial_kernel(synthetic_embeddings, synthetic_embeddings)
        k_xy = _polynomial_kernel(ref_embeddings, synthetic_embeddings)

        term2 = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
        term3 = 2 * k_xy.sum() / (m * n)

        kid = term1 + term2 - term3

        # Negate because Optuna maximizes, and lower KID is better
        final_score = -float(kid)

        logger.debug(f"[KID] Calculated KID: {kid:.6f}")
        logger.debug(f"[KID] Final negated score: {final_score:.6f}")
        return final_score
    except Exception as e:
        logger.error(f"Error in KID computation: {e}", exc_info=True)
        return -float('inf')


def compute_ndb_score(synthetic_embeddings: np.ndarray, ref_hist_bins: List[np.ndarray], ref_hist: np.ndarray,
                      alpha: float = 0.05) -> float:
    logger.debug("[NDB] Computing scores...")
    try:
        synth_hist, _ = np.histogramdd(synthetic_embeddings, bins=ref_hist_bins)
        meaningful_bins_indices = np.argwhere((ref_hist > 0) & (synth_hist > 0))

        if meaningful_bins_indices.shape[0] == 0:
            logger.warning("[NDB] No overlapping bins with non-zero counts. Score is 0.")
            return 0.0

        statistically_different_bins = 0
        total_ref_samples = ref_hist.sum()
        total_synth_samples = synth_hist.sum()

        if total_ref_samples == 0 or total_synth_samples == 0:
            logger.warning("[NDB] One or both histograms are empty. Cannot perform test.")
            return -float('inf')

        for bin_idx_tuple in meaningful_bins_indices:
            bin_idx = tuple(bin_idx_tuple)
            ref_count = ref_hist[bin_idx]
            synth_count = synth_hist[bin_idx]
            contingency_table = np.array([[ref_count, total_ref_samples - ref_count],
                                          [synth_count, total_synth_samples - synth_count]])
            if np.any(np.sum(contingency_table, axis=0) == 0) or np.any(np.sum(contingency_table, axis=1) == 0):
                continue
            try:
                _, p_value, _, _ = chi2_contingency(contingency_table)
                if p_value < alpha:
                    statistically_different_bins += 1
            except ValueError:
                continue

        final_score = -float(statistically_different_bins)
        logger.debug(f"[NDB] Final negated score: {final_score:.6f}")
        return final_score
    except Exception as e:
        logger.error(f"Error in NDB score computation: {e}", exc_info=True)
        return -float('inf')


def compute_js_divergence_score(synthetic_embeddings: np.ndarray, ref_hist_bins: List[np.ndarray],
                                ref_prob: np.ndarray) -> float:
    logger.debug("[JSD] Computing scores...")
    try:
        synth_hist, _ = np.histogramdd(synthetic_embeddings, bins=ref_hist_bins)
        if synth_hist.sum() == 0:
            logger.warning("[JSD] Synthetic histogram is empty. Returning -inf.")
            return -float('inf')

        synth_prob = (synth_hist / synth_hist.sum()).flatten()
        synth_prob[synth_prob == 0] = 1e-10

        if ref_prob.shape != synth_prob.shape:
            logger.error(
                f"[JSD] Probability distribution shapes mismatch. Ref: {ref_prob.shape}, Synth: {synth_prob.shape}")
            return -float('inf')

        jsd = jensenshannon(ref_prob, synth_prob)
        final_score = -jsd
        logger.debug(f"[JSD] Final negated score: {final_score:.6f}")
        return final_score
    except Exception as e:
        logger.error(f"Error in JSD score computation: {e}", exc_info=True)
        return -float('inf')


def precompute_matric_args(tuning_cfg: TuningConfig, ref_embeddings: np.ndarray):
    """
    Pre-computes values based on the selected metric, to avoid redundant calculations
    during multiple trial evaluations.
    """
    logger.info(f"Pre-computing values for metric: '{tuning_cfg.similarity_metric}'")
    precomputed_args = {}
    metric = getattr(tuning_cfg, 'similarity_metric', 'cosine')

    if ref_embeddings.shape[0] == 0:
        logger.error("Reference embeddings are empty during pre-computation. Cannot pre-compute metric arguments.")
        return precomputed_args

    try:
        if metric == 'mahalanobis':
            precomputed_args['ref_mean'] = np.mean(ref_embeddings, axis=0)
            # Add a small regularization for stability if covariance is ill-conditioned
            cov_matrix = np.cov(ref_embeddings, rowvar=False)
            precomputed_args['ref_inv_cov'] = np.linalg.pinv(cov_matrix)
            logger.info("Pre-computed mean and inverse covariance matrix for Mahalanobis.")

        elif metric == 'fid':
            precomputed_args['ref_mean'] = np.mean(ref_embeddings, axis=0)
            precomputed_args['ref_cov'] = np.cov(ref_embeddings, rowvar=False)
            logger.info("Pre-computed mean and covariance matrix for FID.")

        elif metric == 'kid':
            m = ref_embeddings.shape[0]
            if m > 1:
                k_xx = _polynomial_kernel(ref_embeddings, ref_embeddings)
                precomputed_args['ref_kid_term1'] = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
                logger.info("Pre-computed reference kernel matrix term for KID.")
            else:
                logger.warning("Cannot pre-compute KID term, not enough reference samples.")

        elif metric in ['ndb', 'jsd']:
            num_bins = tuning_cfg.num_hist_bins
            hist, bins = np.histogramdd(ref_embeddings, bins=num_bins)
            precomputed_args['ref_hist_bins'] = bins
            logger.info(f"Pre-computed histogram bins for '{metric}'.")

            if metric == 'ndb':
                precomputed_args['ref_hist'] = hist
                logger.info("Pre-computed reference histogram for NDB.")
            if metric == 'jsd':
                if hist.sum() > 0:
                    prob = (hist / hist.sum()).flatten()
                    prob[prob == 0] = 1e-10
                    precomputed_args['ref_prob'] = prob
                    logger.info("Pre-computed reference probability distribution for JSD.")
                else:
                    logger.warning("Reference histogram sum is zero during JSD pre-computation.")
        else:
            logger.debug(f"No specific pre-computation needed for metric '{metric}'.")
    except Exception as e:
        logger.error(f"Error during pre-computation for metric '{metric}': {e}", exc_info=True)
        # It might be best to clear precomputed_args here to force on-the-fly computation or failure
        precomputed_args = {}
    return precomputed_args