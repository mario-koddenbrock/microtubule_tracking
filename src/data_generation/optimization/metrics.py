from typing import Optional

import numpy as np
from scipy.spatial.distance import mahalanobis, jensenshannon
from scipy.stats import chi2_contingency
from sklearn.metrics.pairwise import cosine_similarity

from config.tuning import TuningConfig


def similarity(
    tuning_cfg: TuningConfig,
    ref_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    # --- Optional pre-computed values for optimization ---
    ref_mean: Optional[np.ndarray] = None,
    ref_inv_cov: Optional[np.ndarray] = None,
    ref_hist_bins: Optional[list] = None,
    ref_hist: Optional[np.ndarray] = None,
    ref_prob: Optional[np.ndarray] = None,
) -> float:
    """
    Core evaluation logic: Computes similarity using the specified metric.
    """
    similarity_metric = getattr(tuning_cfg, 'similarity_metric', 'cosine')
    print(f"\n--- [METRIC START] ---")
    print(f"Metric: '{similarity_metric}'")
    print(f"Ref Embeddings Shape: {ref_embeddings.shape}, Min/Max: {ref_embeddings.min():.4f}/{ref_embeddings.max():.4f}")
    print(f"Synth Embeddings Shape: {synthetic_embeddings.shape}, Min/Max: {synthetic_embeddings.min():.4f}/{synthetic_embeddings.max():.4f}")


    # On-the-fly computation for one-off calls
    if similarity_metric == "mahalanobis":
        if ref_mean is None:
            print("[METRIC INFO] Computing ref_mean on the fly")
            ref_mean = np.mean(ref_embeddings, axis=0)
        if ref_inv_cov is None:
            print("[METRIC INFO] Computing ref_inv_cov on the fly")
            ref_inv_cov = np.linalg.pinv(np.cov(ref_embeddings, rowvar=False))
        score = compute_mahalanobis_score(synthetic_embeddings, ref_mean, ref_inv_cov)

    elif similarity_metric == "cosine":
        score = compute_cosine_score(synthetic_embeddings, ref_embeddings)

    elif similarity_metric == "ndb":
        num_bins = getattr(tuning_cfg, 'num_hist_bins', 10)
        if ref_hist_bins is None:
            print(f"[METRIC INFO] Computing histogram bins (n={num_bins}) on the fly")
            joint_embeddings = np.vstack([ref_embeddings, synthetic_embeddings])
            _, ref_hist_bins = np.histogramdd(joint_embeddings, bins=num_bins)
        if ref_hist is None:
            print("[METRIC INFO] Computing ref_hist on the fly")
            ref_hist, _ = np.histogramdd(ref_embeddings, bins=ref_hist_bins)
        score = compute_ndb_score(synthetic_embeddings, ref_hist_bins=ref_hist_bins, ref_hist=ref_hist)

    elif similarity_metric == "jsd":
        num_bins = getattr(tuning_cfg, 'num_hist_bins', 10)
        if ref_hist_bins is None:
            print(f"[METRIC INFO] Computing histogram bins (n={num_bins}) on the fly")
            joint_embeddings = np.vstack([ref_embeddings, synthetic_embeddings])
            _, ref_hist_bins = np.histogramdd(joint_embeddings, bins=num_bins)
        if ref_prob is None:
            print("[METRIC INFO] Computing ref_prob on the fly")
            hist, _ = np.histogramdd(ref_embeddings, bins=ref_hist_bins)
            ref_prob = (hist / hist.sum()).flatten()
            ref_prob[ref_prob == 0] = 1e-10
        score = compute_js_divergence_score(synthetic_embeddings, ref_hist_bins=ref_hist_bins, ref_prob=ref_prob)
    else:
        raise ValueError(f"Unknown similarity metric: '{similarity_metric}'.")

    print(f"Final Score for Trial: {score:.6f}")
    print(f"--- [METRIC END] ---\n")
    return score


def compute_cosine_score(synthetic_embeddings: np.ndarray, ref_embeddings: np.ndarray) -> float:
    print("[DEBUG-COSINE] Computing scores...")
    frame_scores = [np.max(cosine_similarity(emb.reshape(1, -1), ref_embeddings)[0]) for emb in synthetic_embeddings]
    print(f"[DEBUG-COSINE] Individual frame scores (first 5): {np.array(frame_scores[:5])}")
    mean_score = float(np.mean(frame_scores))
    print(f"[DEBUG-COSINE] Mean score: {mean_score}")
    return mean_score


def compute_mahalanobis_score(synthetic_embeddings: np.ndarray, ref_mean: np.ndarray, ref_inv_cov: np.ndarray) -> float:
    print("[DEBUG-MAHALANOBIS] Computing distances...")
    print(f"[DEBUG-MAHALANOBIS] ref_mean stats: Min={ref_mean.min():.4f}, Max={ref_mean.max():.4f}, Mean={ref_mean.mean():.4f}")
    print(f"[DEBUG-MAHALANOBIS] ref_inv_cov stats: Min={ref_inv_cov.min():.4f}, Max={ref_inv_cov.max():.4f}, Mean={ref_inv_cov.mean():.4f}")

    distances = []
    for i, emb in enumerate(synthetic_embeddings):
        u = emb.squeeze()
        delta = u - ref_mean
        d_squared = np.dot(np.dot(delta, ref_inv_cov), delta.T)
        safe_d_squared = np.maximum(0, d_squared)
        dist = np.sqrt(safe_d_squared)
        if i < 5: # Print details for the first 5 embeddings
             print(f"  [Emb {i}] d_squared={d_squared:.6f}, safe_d_squared={safe_d_squared:.6f}, distance={dist:.6f}")
        distances.append(dist)

    mean_distance = np.mean(distances)
    final_score = -float(mean_distance)
    print(f"[DEBUG-MAHALANOBIS] Mean distance: {mean_distance:.6f}")
    print(f"[DEBUG-MAHALANOBIS] Final negated score: {final_score:.6f}")
    return final_score


def compute_ndb_score(synthetic_embeddings: np.ndarray, ref_hist_bins: list, ref_hist: np.ndarray, alpha: float = 0.05) -> float:
    print("[DEBUG-NDB] Computing scores...")
    synth_hist, _ = np.histogramdd(synthetic_embeddings, bins=ref_hist_bins)
    print(f"[DEBUG-NDB] Ref hist sum: {ref_hist.sum()}, Synth hist sum: {synth_hist.sum()}")
    print(f"[DEBUG-NDB] Ref hist non-zero bins: {np.count_nonzero(ref_hist)}, Synth hist non-zero bins: {np.count_nonzero(synth_hist)}")

    meaningful_bins = np.argwhere((ref_hist > 0) & (synth_hist > 0))
    print(f"[DEBUG-NDB] Number of overlapping (meaningful) bins: {meaningful_bins.shape[0]}")

    if meaningful_bins.shape[0] == 0:
        print("[DEBUG-NDB] No overlapping bins found. Returning score 0.0.")
        return 0.0

    statistically_different_bins = 0
    total_ref_samples = ref_hist.sum()
    total_synth_samples = synth_hist.sum()

    for bin_idx_tuple in meaningful_bins:
        bin_idx = tuple(bin_idx_tuple)
        ref_count = ref_hist[bin_idx]
        synth_count = synth_hist[bin_idx]
        contingency_table = np.array([[ref_count, total_ref_samples - ref_count],
                                      [synth_count, total_synth_samples - synth_count]])
        _, p_value, _, _ = chi2_contingency(contingency_table)
        if p_value < alpha:
            statistically_different_bins += 1

    print(f"[DEBUG-NDB] Number of statistically different bins: {statistically_different_bins}")
    final_score = -float(statistically_different_bins)
    print(f"[DEBUG-NDB] Final negated score: {final_score}")
    return final_score


def compute_js_divergence_score(synthetic_embeddings: np.ndarray, ref_hist_bins: list, ref_prob: np.ndarray) -> float:
    print("[DEBUG-JSD] Computing scores...")
    synth_hist, _ = np.histogramdd(synthetic_embeddings, bins=ref_hist_bins)
    print(f"[DEBUG-JSD] Synth hist sum: {synth_hist.sum()}, non-zero bins: {np.count_nonzero(synth_hist)}")

    # Check if the synthetic histogram is all zeros
    if synth_hist.sum() == 0:
        print("[DEBUG-JSD] WARNING: Synthetic histogram is empty. Score might be unpredictable.")
        return -1.0 # Return a bad score

    synth_prob = (synth_hist / synth_hist.sum()).flatten()
    synth_prob[synth_prob == 0] = 1e-10

    print(f"[DEBUG-JSD] Ref prob shape: {ref_prob.shape}, Synth prob shape: {synth_prob.shape}")
    print(f"[DEBUG-JSD] Ref prob sum: {ref_prob.sum():.4f}, Synth prob sum: {synth_prob.sum():.4f}")

    jsd = jensenshannon(ref_prob, synth_prob)
    print(f"[DEBUG-JSD] Jensen-Shannon Distance: {jsd:.6f}")
    final_score = -jsd
    print(f"[DEBUG-JSD] Final negated score: {final_score:.6f}")
    return final_score



def precompute_matric_args(tuning_cfg, ref_embeddings):
    # --- Pre-compute values based on the selected metric ---
    precomputed_args = {}
    metric = getattr(tuning_cfg, 'similarity_metric', 'cosine')
    print(f"Pre-computing values for metric: '{metric}'")
    if metric == 'mahalanobis':
        precomputed_args['ref_mean'] = np.mean(ref_embeddings, axis=0)
        precomputed_args['ref_inv_cov'] = np.linalg.pinv(np.cov(ref_embeddings, rowvar=False))
        print("Pre-computed mean and inverse covariance matrix.")
        print(f"  Mean shape: {precomputed_args['ref_mean'].shape}")
        print(f"  Inv Cov shape: {precomputed_args['ref_inv_cov'].shape}")
    elif metric in ['ndb', 'jsd']:
        num_bins = tuning_cfg.num_hist_bins
        hist, bins = np.histogramdd(ref_embeddings, bins=num_bins)
        precomputed_args['ref_hist_bins'] = bins
        print(f"Pre-computed histogram bins (num_bins={num_bins}).")

        if metric == 'ndb':
            precomputed_args['ref_hist'] = hist
            print("Pre-computed reference histogram for NDB.")
        if metric == 'jsd':
            prob = (hist / hist.sum()).flatten()
            prob[prob == 0] = 1e-10
            precomputed_args['ref_prob'] = prob
            print("Pre-computed reference probability distribution for JSD.")
    return precomputed_args