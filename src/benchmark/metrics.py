# src/benchmark/metrics.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, entropy
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries


def _compute_iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """
    Computes the IoU matrix between predicted and ground truth masks.
    Args:
        pred_masks: (N, H, W) bool array of predicted masks.
        gt_masks: (M, H, W) bool array of ground truth masks.
    Returns:
        (M, N) np.ndarray of IoU values.
    """
    num_gt, num_pred = len(gt_masks), len(pred_masks)
    if num_pred == 0 or num_gt == 0:
        return np.zeros((num_gt, num_pred))

    # Vectorized computation of intersection and union
    intersection = np.einsum('mhw,nhw->mn', gt_masks, pred_masks)
    union = gt_masks.sum(axis=(1, 2))[:, None] + pred_masks.sum(axis=(1, 2))[None, :] - intersection

    iou_matrix = intersection / np.maximum(union, 1e-6)
    return iou_matrix


def _get_matches(iou_matrix: np.ndarray, iou_threshold: float):
    """
    Finds matches between GT and predicted masks using the Hungarian algorithm.
    """
    if iou_matrix.shape[0] == 0 or iou_matrix.shape[1] == 0:
        return [], [], []

    # Use Hungarian algorithm to find optimal assignment
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    matches, unmatched_preds, unmatched_gts = [], [], []

    # Filter matches based on IoU threshold
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        if iou_matrix[gt_idx, pred_idx] >= iou_threshold:
            matches.append((gt_idx, pred_idx))

    # Identify unmatched predictions and ground truths
    matched_preds = {p for _, p in matches}
    unmatched_preds = set(range(iou_matrix.shape[1])) - matched_preds

    matched_gts = {g for g, _ in matches}
    unmatched_gts = set(range(iou_matrix.shape[0])) - matched_gts

    return matches, list(unmatched_preds), list(unmatched_gts)


def _boundary_f1(pred_mask, gt_mask, bound_th=2):
    """
    Computes the Boundary F1 score between a predicted and ground truth mask.
    """
    gt_boundary = find_boundaries(gt_mask, mode='inner')
    pred_boundary = find_boundaries(pred_mask, mode='inner')

    from scipy.ndimage import distance_transform_edt

    # Handle cases where a boundary is empty
    if not np.any(gt_boundary) or not np.any(pred_boundary):
        return 0.0 if np.any(gt_boundary) or np.any(pred_boundary) else 1.0

    dist_gt_to_pred = distance_transform_edt(np.logical_not(pred_boundary))
    dist_pred_to_gt = distance_transform_edt(np.logical_not(gt_boundary))

    precision = np.count_nonzero(dist_gt_to_pred[gt_boundary] <= bound_th) / np.count_nonzero(gt_boundary)
    recall = np.count_nonzero(dist_pred_to_gt[pred_boundary] <= bound_th) / np.count_nonzero(pred_boundary)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_segmentation_metrics(pred_masks: np.ndarray, gt_masks: np.ndarray, iou_thresholds=(0.5, 0.75)):
    """
    Calculates AP, F1, and Boundary F1 scores.
    Args:
        pred_masks: (N, H, W) predicted instance masks.
        gt_masks: (M, H, W) ground truth instance masks.
    Returns:
        A dictionary with calculated metrics.
    """
    pred_masks = pred_masks.astype(bool)
    gt_masks = gt_masks.astype(bool)

    iou_matrix = _compute_iou_matrix(gt_masks, pred_masks)

    metrics = {}

    # Calculate AP (average over IoU thresholds from 0.5 to 0.95)
    precisions = []
    for thresh in np.arange(0.5, 1.0, 0.05):
        matches, unmatched_preds, _ = _get_matches(iou_matrix, thresh)
        tp = len(matches)
        fp = len(unmatched_preds)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    metrics['AP'] = np.mean(precisions) if precisions else 0.0

    # Calculate F1 and Boundary F1 for specified thresholds
    for thresh in iou_thresholds:
        matches, unmatched_preds, unmatched_gts = _get_matches(iou_matrix, thresh)

        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)

        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        metrics[f'F1@{thresh:.2f}'] = f1

        bf1_scores = [_boundary_f1(pred_masks[p_idx], gt_masks[g_idx]) for g_idx, p_idx in matches]
        metrics[f'BF1@{thresh:.2f}'] = np.mean(bf1_scores) if bf1_scores else 0.0

    return metrics


def _get_length_distribution(masks: np.ndarray) -> np.ndarray:
    """Calculates the length of each instance mask after skeletonization."""
    if masks.size == 0:
        return np.array([])
    lengths = [np.sum(skeletonize(mask)) for mask in masks]
    return np.array(lengths)


def calculate_downstream_metrics(pred_masks: np.ndarray, gt_masks: np.ndarray):
    """
    Calculates KS and KL metrics for the length distribution of instances.
    Args:
        pred_masks: (N, H, W) predicted instance masks.
        gt_masks: (M, H, W) ground truth instance masks.
    Returns:
        A dictionary with calculated metrics for length distribution.
    """
    pred_lengths = _get_length_distribution(pred_masks)
    gt_lengths = _get_length_distribution(gt_masks)

    metrics = {'Length_KS': np.nan, 'Length_KL': np.nan}

    if len(pred_lengths) == 0 or len(gt_lengths) == 0:
        return metrics  # Not enough data to compare

    # Kolmogorov-Smirnov statistic
    ks_stat, _ = ks_2samp(pred_lengths, gt_lengths)
    metrics['Length_KS'] = ks_stat

    # Kullback-Leibler divergence
    min_len = min(pred_lengths.min(), gt_lengths.min())
    max_len = max(pred_lengths.max(), gt_lengths.max())
    num_bins = max(10, int(np.sqrt(len(gt_lengths)))) # Freedman-Diaconis or similar could be better
    bins = np.linspace(min_len, max_len, num_bins)

    pred_hist, _ = np.histogram(pred_lengths, bins=bins, density=True)
    gt_hist, _ = np.histogram(gt_lengths, bins=bins, density=True)

    # Add a small epsilon to avoid division by zero in KL divergence
    epsilon = 1e-10
    pred_dist = pred_hist + epsilon
    gt_dist = gt_hist + epsilon

    pred_dist /= np.sum(pred_dist)
    gt_dist /= np.sum(gt_dist)

    metrics['Length_KL'] = entropy(pk=pred_dist, qk=gt_dist)

    return metrics