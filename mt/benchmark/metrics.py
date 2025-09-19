import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, entropy, wasserstein_distance
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries


# -------------------------
# Utilities
# -------------------------
def _as_instance_stack(mask_or_stack: np.ndarray) -> np.ndarray:
    """
    Accept either a labeled mask (H, W) with background=0, or a stack (N, H, W).
    Return a boolean stack (N, H, W) of instance masks.
    """
    arr = np.asarray(mask_or_stack)
    if arr.ndim == 3:
        return arr.astype(bool)
    if arr.ndim != 2:
        raise ValueError(f"Expected (H,W) labeled mask or (N,H,W) stack, got {arr.shape}")
    ids = np.unique(arr)
    ids = ids[ids != 0]
    if ids.size == 0:
        return np.zeros((0, arr.shape[0], arr.shape[1]), dtype=bool)
    return np.stack([arr == i for i in ids], axis=0).astype(bool)


def _compute_iou_matrix(
    gt_instance_masks: np.ndarray, pred_instance_masks: np.ndarray
) -> np.ndarray:
    """
    Compute IoU matrix between GT and predicted instance masks.
    Args:
        gt_instance_masks  : (M, H, W) bool
        pred_instance_masks: (N, H, W) bool
    Returns:
        (M, N) float IoU matrix
    """
    M, N = len(gt_instance_masks), len(pred_instance_masks)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=float)

    # Ensure boolean, then use int64 accumulation to avoid overflow
    gm = gt_instance_masks.astype(bool)
    pm = pred_instance_masks.astype(bool)

    # intersections: (M, N)
    inter = (gm[:, None, :, :] & pm[None, :, :, :]).sum(axis=(2, 3), dtype=np.int64)

    gt_areas = gm.sum(axis=(1, 2), dtype=np.int64)[:, None]  # (M, 1)
    pr_areas = pm.sum(axis=(1, 2), dtype=np.int64)[None, :]  # (1, N)
    union = gt_areas + pr_areas - inter

    # Safe division to float
    return inter / np.maximum(union, 1)


def _get_matches(
    iou_matrix: np.ndarray, iou_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Hungarian assignment on IoU to get GT<->Pred matches above threshold.
    Returns:
        matches: list of (gt_idx, pred_idx)
        unmatched_preds: list of pred indices
        unmatched_gts  : list of gt indices
    """
    M, N = iou_matrix.shape
    if M == 0 or N == 0:
        return [], list(range(N)), list(range(M))

    gt_idx, pr_idx = linear_sum_assignment(-iou_matrix)
    matches = [(g, p) for g, p in zip(gt_idx, pr_idx) if iou_matrix[g, p] >= iou_threshold]

    matched_g = {g for g, _ in matches}
    matched_p = {p for _, p in matches}

    unmatched_g = [g for g in range(M) if g not in matched_g]
    unmatched_p = [p for p in range(N) if p not in matched_p]
    return matches, unmatched_p, unmatched_g


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    return find_boundaries(mask.astype(bool), mode="outer")


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Directed boundary distances A→B and B→A via EDT."""
    from scipy.ndimage import distance_transform_edt

    Ba = _extract_boundary(mask_a)
    Bb = _extract_boundary(mask_b)
    if Ba.sum() == 0 or Bb.sum() == 0:
        return np.array([]), np.array([])
    dtB = distance_transform_edt(~Bb)
    dtA = distance_transform_edt(~Ba)
    return dtB[Ba], dtA[Bb]


def _hausdorff_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    d_ab, d_ba = _surface_distances(mask_a, mask_b)
    if d_ab.size == 0 or d_ba.size == 0:
        return np.nan
    return float(max(d_ab.max(), d_ba.max()))


def _assd(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    d_ab, d_ba = _surface_distances(mask_a, mask_b)
    if d_ab.size == 0 or d_ba.size == 0:
        return np.nan
    return float(0.5 * (d_ab.mean() + d_ba.mean()))


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    A = mask_a.astype(bool)
    B = mask_b.astype(bool)
    inter = np.logical_and(A, B).sum()
    denom = A.sum() + B.sum()
    return 1.0 if denom == 0 else float((2.0 * inter) / (denom + 1e-12))


def _boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, bound_th: int = 2) -> float:
    """Boundary F1 between two masks using pixel tolerance bound_th."""
    from scipy.ndimage import distance_transform_edt

    gt_boundary = find_boundaries(gt_mask, mode="inner")
    pr_boundary = find_boundaries(pred_mask, mode="inner")

    if not np.any(gt_boundary) or not np.any(pr_boundary):
        return 0.0 if (np.any(gt_boundary) or np.any(pr_boundary)) else 1.0

    dist_gt_to_pr = distance_transform_edt(~pr_boundary)
    dist_pr_to_gt = distance_transform_edt(~gt_boundary)

    precision = (dist_gt_to_pr[gt_boundary] <= bound_th).mean()
    recall = (dist_pr_to_gt[pr_boundary] <= bound_th).mean()
    return (
        0.0 if (precision + recall) == 0 else float(2 * precision * recall / (precision + recall))
    )


def _panoptic_quality(
    iou_matrix: np.ndarray, iou_threshold: float = 0.5
) -> Tuple[float, float, float]:
    """Compute PQ, SQ, DQ given stacks and IoU matrix."""
    matches, unmatched_preds, unmatched_gts = _get_matches(iou_matrix, iou_threshold)
    tp, fp, fn = len(matches), len(unmatched_preds), len(unmatched_gts)
    if tp + fp + fn == 0:
        return 0.0, 0.0, 0.0
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    if tp == 0:
        sq = 0.0
    else:
        ious = [float(iou_matrix[g, p]) for g, p in matches]
        sq = float(np.mean(ious))
    pq = dq * sq
    return pq, sq, dq


def _count_error(
    gt_instance_masks: np.ndarray, pred_instance_masks: np.ndarray
) -> Tuple[int, float]:
    gt_count = int(gt_instance_masks.shape[0])
    pred_count = int(pred_instance_masks.shape[0])
    abs_err = abs(pred_count - gt_count)
    rel_err = abs_err / (gt_count + 1e-6)
    return abs_err, float(rel_err)


def _get_length_distribution(masks: np.ndarray) -> np.ndarray:
    """Skeleton length per instance (proxy for filament length)."""
    if masks.size == 0:
        return np.array([])
    return np.array([np.sum(skeletonize(m)) for m in masks], dtype=float)


def _get_curvature_distribution(masks: np.ndarray) -> np.ndarray:
    """
    Computes the mean curvature for the skeleton of each mask.
    Returns an array of mean curvatures, one for each mask.
    """
    if masks.size == 0:
        return np.array([])

    curvatures = []
    for mask in masks:
        skeleton = skeletonize(mask)
        coords = np.argwhere(skeleton)
        if len(coords) < 5:  # Need enough points to calculate curvature
            curvatures.append(np.nan)
            continue

        # Simple ordering of points for non-branching skeletons
        # This is a simplification and may not work for complex skeletons
        from scipy.spatial.distance import cdist

        ordered_coords = [coords[0]]
        remaining_coords = list(coords[1:])

        current_coord = coords[0]
        while remaining_coords:
            distances = cdist([current_coord], remaining_coords)
            nearest_idx = np.argmin(distances)
            current_coord = remaining_coords.pop(nearest_idx)
            ordered_coords.append(current_coord)

        ordered_coords = np.array(ordered_coords)

        # Using np.gradient to estimate derivatives
        dx = np.gradient(ordered_coords[:, 1].astype(float))
        dy = np.gradient(ordered_coords[:, 0].astype(float))
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)

        # Curvature formula: k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * d2y - dy * d2x)
        denominator = (dx**2 + dy**2) ** 1.5

        # Avoid division by zero for straight segments
        instance_curvatures = np.divide(numerator, denominator, where=denominator != 0)

        mean_curvature = np.nanmean(instance_curvatures)
        curvatures.append(mean_curvature)

    return np.array(curvatures, dtype=float)


# -------------------------
# Public metrics
# -------------------------
def calculate_segmentation_metrics(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    iou_thresholds: Tuple[float, ...] = (0.5, 0.75, 0.9),
) -> Dict[str, float]:
    """
    Compute instance segmentation metrics.

    Notes:
      - 'AP50-95' here is NOT true Average Precision (no score ranking). It is the
        mean of precision = TP/(TP+FP) measured at IoU thresholds 0.50:0.05:0.95.
    """
    pred_instance_masks = _as_instance_stack(pred_masks)
    gt_instance_masks = _as_instance_stack(gt_masks)

    iou_matrix = _compute_iou_matrix(gt_instance_masks, pred_instance_masks)
    metrics: Dict[str, float] = {}

    # “AP50–95” (mean precision across IoU thresholds)
    precisions: List[float] = []
    for t in np.arange(0.5, 1.0, 0.05):
        matches, unmatched_preds, _ = _get_matches(iou_matrix, float(t))
        tp = len(matches)
        fp = len(unmatched_preds)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(precision)
        if np.isclose(t, 0.5) or np.isclose(t, 0.75) or np.isclose(t, 0.9):
            metrics[f"AP@{t:.2f}"] = float(precision)

    metrics["AP50-95"] = float(np.mean(precisions) if precisions else 0.0)
    metrics["AP"] = metrics["AP50-95"]  # alias for compatibility

    # Thresholded metrics
    for thresh in iou_thresholds:
        matches, unmatched_preds, unmatched_gts = _get_matches(iou_matrix, float(thresh))
        tp, fp, fn = len(matches), len(unmatched_preds), len(unmatched_gts)

        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        metrics[f"F1@{thresh:.2f}"] = float(f1)

        # Pairwise metrics only over matched pairs
        if tp > 0:
            bf1 = np.mean(
                [_boundary_f1(pred_instance_masks[p], gt_instance_masks[g]) for g, p in matches]
            )
            dice = np.mean(
                [_dice(pred_instance_masks[p], gt_instance_masks[g]) for g, p in matches]
            )
            hd = np.nanmean(
                [
                    _hausdorff_distance(pred_instance_masks[p], gt_instance_masks[g])
                    for g, p in matches
                ]
            )
            assd = np.nanmean(
                [_assd(pred_instance_masks[p], gt_instance_masks[g]) for g, p in matches]
            )
        else:
            bf1 = 0.0
            dice = 0.0
            hd = np.nan
            assd = np.nan

        metrics[f"BF1@{thresh:.2f}"] = float(bf1)
        metrics[f"Dice@{thresh:.2f}"] = float(dice)
        metrics[f"Hausdorff@{thresh:.2f}"] = float(hd) if not np.isnan(hd) else np.nan
        metrics[f"ASSD@{thresh:.2f}"] = float(assd) if not np.isnan(assd) else np.nan

        pq, sq, dq = _panoptic_quality(iou_matrix, iou_threshold=float(thresh))
        metrics[f"PQ@{thresh:.2f}"] = float(pq)
        metrics[f"SQ@{thresh:.2f}"] = float(sq)
        metrics[f"DQ@{thresh:.2f}"] = float(dq)

    # Counting
    abs_err, rel_err = _count_error(gt_instance_masks, pred_instance_masks)
    metrics["CountAbsErr"] = float(abs_err)
    metrics["CountRelErr"] = float(rel_err)

    # IoU
    if gt_instance_masks.shape[0] > 0 and pred_instance_masks.shape[0] > 0:
        ious = iou_matrix.max(axis=1)  # best IoU for each GT
        metrics["IoU_mean"] = float(np.mean(ious))
        metrics["IoU_median"] = float(np.median(ious))
    else:
        metrics["IoU_mean"] = np.nan
        metrics["IoU_median"] = np.nan

    return metrics


def calculate_downstream_metrics(pred_masks: np.ndarray, gt_masks: np.ndarray) -> Dict[str, float]:
    """
    Compare simple downstream distributions (length, curvature) and counts.
    Returns:
        {
          'Length_KS', 'Length_KL', 'Length_EMD',
          'Curvature_KS', 'Curvature_KL', 'Curvature_EMD',
          'Count_pred', 'Count_gt'
        }
    """
    pred_instance_masks = _as_instance_stack(pred_masks)
    gt_instance_masks = _as_instance_stack(gt_masks)

    out: Dict[str, float] = {
        "Length_KS": np.nan,
        "Length_KL": np.nan,
        "Length_EMD": np.nan,
        "Curvature_KS": np.nan,
        "Curvature_KL": np.nan,
        "Curvature_EMD": np.nan,
    }

    # Length distribution
    pred_lengths = _get_length_distribution(pred_instance_masks)
    gt_lengths = _get_length_distribution(gt_instance_masks)

    if pred_lengths.size > 0 and gt_lengths.size > 0:
        ks_stat, _ = ks_2samp(pred_lengths, gt_lengths)
        out["Length_KS"] = float(ks_stat)

        min_len = float(min(pred_lengths.min(), gt_lengths.min()))
        max_len = float(max(pred_lengths.max(), gt_lengths.max()))
        if max_len > min_len:
            num_bins = max(10, int(np.sqrt(len(gt_lengths))))
            bins = np.linspace(min_len, max_len, num_bins)
            pred_hist, _ = np.histogram(pred_lengths, bins=bins, density=True)
            gt_hist, _ = np.histogram(gt_lengths, bins=bins, density=True)
            eps = 1e-10
            pred_dist = pred_hist + eps
            gt_dist = gt_hist + eps
            pred_dist /= pred_dist.sum()
            gt_dist /= gt_dist.sum()
            out["Length_KL"] = float(entropy(pk=pred_dist, qk=gt_dist))

        out["Length_EMD"] = float(wasserstein_distance(pred_lengths, gt_lengths))

    # Curvature distribution
    pred_curvatures = _get_curvature_distribution(pred_instance_masks)
    gt_curvatures = _get_curvature_distribution(gt_instance_masks)

    pred_curvatures = pred_curvatures[~np.isnan(pred_curvatures)]
    gt_curvatures = gt_curvatures[~np.isnan(gt_curvatures)]

    if pred_curvatures.size > 0 and gt_curvatures.size > 0:
        ks_stat, _ = ks_2samp(pred_curvatures, gt_curvatures)
        out["Curvature_KS"] = float(ks_stat)

        min_curv = float(min(pred_curvatures.min(), gt_curvatures.min()))
        max_curv = float(max(pred_curvatures.max(), gt_curvatures.max()))
        if max_curv > min_curv:
            num_bins = max(10, int(np.sqrt(len(gt_curvatures))))
            bins = np.linspace(min_curv, max_curv, num_bins)
            pred_hist, _ = np.histogram(pred_curvatures, bins=bins, density=True)
            gt_hist, _ = np.histogram(gt_curvatures, bins=bins, density=True)
            eps = 1e-10
            pred_dist = pred_hist + eps
            gt_dist = gt_hist + eps
            pred_dist /= pred_dist.sum()
            gt_dist /= gt_dist.sum()
            out["Curvature_KL"] = float(entropy(pk=pred_dist, qk=gt_dist))

        out["Curvature_EMD"] = float(wasserstein_distance(pred_curvatures, gt_curvatures))

    # Simple counts
    out["Count_pred"] = float(pred_instance_masks.shape[0])
    out["Count_gt"] = float(gt_instance_masks.shape[0])

    return out
