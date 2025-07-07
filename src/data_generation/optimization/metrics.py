import logging
from typing import Optional, List

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from sklearn.metrics.pairwise import cosine_similarity

from config.tuning import TuningConfig

logger = logging.getLogger(f"mt.{__name__}")


def similarity(
        tuning_cfg: TuningConfig,
        ref_embeddings: np.ndarray,
        synthetic_embeddings: np.ndarray,
        # --- Optional pre-computed values for optimization ---
        ref_mean: Optional[np.ndarray] = None,
        ref_inv_cov: Optional[np.ndarray] = None,
        ref_hist_bins: Optional[List[np.ndarray]] = None,  # Changed to List[np.ndarray] based on np.histogramdd output
        ref_hist: Optional[np.ndarray] = None,
        ref_prob: Optional[np.ndarray] = None,
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
    logger.debug(
        f"[MAHALANOBIS] ref_mean stats: Min={ref_mean.min():.4f}, Max={ref_mean.max():.4f}, Mean={ref_mean.mean():.4f}")
    logger.debug(
        f"[MAHALANOBIS] ref_inv_cov stats: Min={ref_inv_cov.min():.4f}, Max={ref_inv_cov.max():.4f}, Mean={ref_inv_cov.mean():.4f}")

    distances = []
    for i, emb in enumerate(synthetic_embeddings):
        try:
            u = emb.squeeze()
            if u.ndim != 1:
                logger.warning(
                    f"Embedding {i} has unexpected dimensions after squeeze: {u.shape}. Attempting to flatten.")
                u = u.flatten()

            delta = u - ref_mean

            # Ensure dimensions match for dot products
            if delta.shape[0] != ref_inv_cov.shape[0] or ref_inv_cov.shape[1] != delta.shape[0]:
                logger.error(
                    f"Dimension mismatch for Mahalanobis calculation. Delta shape: {delta.shape}, Inv Cov shape: {ref_inv_cov.shape}. Skipping embedding {i}.")
                distances.append(float('nan'))
                continue

            d_squared = np.dot(np.dot(delta, ref_inv_cov), delta.T)
            safe_d_squared = np.maximum(0, d_squared)  # Ensure non-negative before sqrt
            dist = np.sqrt(safe_d_squared)

            if logger.isEnabledFor(logging.DEBUG) and i < 5:  # Log first 5 for debug
                logger.debug(
                    f"  [Emb {i}] d_squared={d_squared:.6f}, safe_d_squared={safe_d_squared:.6f}, distance={dist:.6f}")
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


def compute_ndb_score(synthetic_embeddings: np.ndarray, ref_hist_bins: List[np.ndarray], ref_hist: np.ndarray,
                      alpha: float = 0.05) -> float:
    logger.debug("[NDB] Computing scores...")
    try:
        synth_hist, _ = np.histogramdd(synthetic_embeddings, bins=ref_hist_bins)
        logger.debug(f"[NDB] Ref hist shape: {ref_hist.shape}, Synth hist shape: {synth_hist.shape}")
        logger.debug(f"[NDB] Ref hist sum: {ref_hist.sum()}, Synth hist sum: {synth_hist.sum()}")
        logger.debug(
            f"[NDB] Ref hist non-zero bins: {np.count_nonzero(ref_hist)}, Synth hist non-zero bins: {np.count_nonzero(synth_hist)}")

        # Find bins where both histograms have non-zero counts
        meaningful_bins_indices = np.argwhere((ref_hist > 0) & (synth_hist > 0))
        logger.debug(f"[NDB] Number of overlapping (meaningful) bins: {meaningful_bins_indices.shape[0]}")

        if meaningful_bins_indices.shape[0] == 0:
            logger.warning(
                "[NDB] No overlapping bins found where both reference and synthetic data have counts. Returning score 0.0 (or -inf if aiming for max).")
            return 0.0  # Or -float('inf') if 0.0 is too good of a score for no overlap

        statistically_different_bins = 0
        total_ref_samples = ref_hist.sum()
        total_synth_samples = synth_hist.sum()

        if total_ref_samples == 0 or total_synth_samples == 0:
            logger.warning(
                f"[NDB] One or both histograms are empty (ref_sum={total_ref_samples}, synth_sum={total_synth_samples}). Cannot perform chi-squared test. Returning -inf.")
            return -float('inf')

        for bin_idx_tuple in meaningful_bins_indices:
            bin_idx = tuple(bin_idx_tuple)
            ref_count = ref_hist[bin_idx]
            synth_count = synth_hist[bin_idx]

            # Chi-squared test needs at least one observation in each cell, and typically > 5 for expected counts
            # For 2x2 table, sums must be > 0.
            if ref_count == 0 and synth_count == 0:  # Should not happen with meaningful_bins_indices, but defensive
                continue

            contingency_table = np.array([[ref_count, total_ref_samples - ref_count],
                                          [synth_count, total_synth_samples - synth_count]])

            # Ensure no rows/columns sum to zero to avoid chi2_contingency errors
            if np.any(np.sum(contingency_table, axis=0) == 0) or np.any(np.sum(contingency_table, axis=1) == 0):
                logger.warning(
                    f"[NDB] Contingency table has zero marginal sum for bin {bin_idx_tuple}. Skipping chi-squared for this bin.")
                continue

            try:
                _, p_value, _, _ = chi2_contingency(contingency_table)
                if p_value < alpha:
                    statistically_different_bins += 1
            except ValueError as e:  # Catch errors from chi2_contingency if input is degenerate
                logger.warning(f"[NDB] Chi-squared test failed for bin {bin_idx_tuple}: {e}. Skipping this bin.")
                continue

        logger.debug(f"[NDB] Number of statistically different bins: {statistically_different_bins}")
        final_score = -float(statistically_different_bins)  # Negate as more different bins is worse
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
        logger.debug(f"[JSD] Synth hist sum: {synth_hist.sum()}, non-zero bins: {np.count_nonzero(synth_hist)}")

        # Check if the synthetic histogram is all zeros
        if synth_hist.sum() == 0:
            logger.warning("[JSD] Synthetic histogram is empty. Jensen-Shannon Divergence will be max. Returning -inf.")
            return -float('inf')

        synth_prob = (synth_hist / synth_hist.sum()).flatten()
        synth_prob[synth_prob == 0] = 1e-10  # Prevent log(0) in JSD

        if ref_prob.shape != synth_prob.shape:
            logger.error(
                f"[JSD] Reference probability ({ref_prob.shape}) and synthetic probability ({synth_prob.shape}) arrays have different shapes. Returning -inf.")
            return -float('inf')

        logger.debug(f"[JSD] Ref prob shape: {ref_prob.shape}, Synth prob shape: {synth_prob.shape}")
        logger.debug(f"[JSD] Ref prob sum: {np.sum(ref_prob):.4f}, Synth prob sum: {np.sum(synth_prob):.4f}")

        # JSD from scipy.spatial.distance returns 0.0 for identical, and larger for more divergence. Max is log(2)
        jsd = jensenshannon(ref_prob, synth_prob)
        logger.debug(f"[JSD] Jensen-Shannon Distance: {jsd:.6f}")
        final_score = -jsd  # Negate JSD as lower divergence (closer to 0) is better
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
        return precomputed_args  # Return empty, downstream will handle

    try:
        if metric == 'mahalanobis':
            precomputed_args['ref_mean'] = np.mean(ref_embeddings, axis=0)
            # Add a small regularization for stability if covariance is ill-conditioned
            cov_matrix = np.cov(ref_embeddings, rowvar=False)
            if cov_matrix.ndim == 0:  # Handle case of single feature or all identical values
                logger.warning(
                    "Covariance matrix is scalar (likely single feature or all identical). Setting inv_cov to identity/small value.")
                precomputed_args['ref_inv_cov'] = np.array([[1.0]])  # or similar, depending on expected dim
            elif np.linalg.det(cov_matrix) == 0:
                logger.warning(
                    "Covariance matrix is singular (determinant is zero). Adding small regularization for pinv stability.")
                cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6  # Add tiny diagonal to make it invertible

            precomputed_args['ref_inv_cov'] = np.linalg.pinv(cov_matrix)
            logger.info("Pre-computed mean and inverse covariance matrix for Mahalanobis.")
            logger.debug(f"  Mean shape: {precomputed_args['ref_mean'].shape}")
            logger.debug(f"  Inv Cov shape: {precomputed_args['ref_inv_cov'].shape}")
        elif metric in ['ndb', 'jsd']:
            num_bins = tuning_cfg.num_hist_bins
            logger.debug(f"Calculating histogram for pre-computation (num_bins={num_bins}).")
            hist, bins = np.histogramdd(ref_embeddings, bins=num_bins)
            precomputed_args['ref_hist_bins'] = bins
            logger.info(f"Pre-computed histogram bins for '{metric}'.")

            if metric == 'ndb':
                precomputed_args['ref_hist'] = hist
                logger.info("Pre-computed reference histogram for NDB.")
            if metric == 'jsd':
                if hist.sum() == 0:
                    logger.warning(
                        "Reference histogram sum is zero during JSD pre-computation. Probability distribution will be empty.")
                    # Do not set ref_prob, let the JSD function handle this as a fatal error
                else:
                    prob = (hist / hist.sum()).flatten()
                    prob[prob == 0] = 1e-10  # Prevent log(0) in JSD
                    precomputed_args['ref_prob'] = prob
                    logger.info("Pre-computed reference probability distribution for JSD.")
        else:
            logger.debug(f"No specific pre-computation needed for metric '{metric}'.")
    except Exception as e:
        logger.error(f"Error during pre-computation for metric '{metric}': {e}", exc_info=True)
        # It might be best to clear precomputed_args here to force on-the-fly computation or failure
        precomputed_args = {}
    return precomputed_args