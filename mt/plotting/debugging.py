import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
import matplotlib as mpl
from PIL import Image

from mt.benchmark.metrics import _as_instance_stack


def plot_gt_pred_overlays(
    img: np.ndarray,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    *,
    boundary: bool = True,  # draw only boundaries (True) or fill regions (False)
    thickness: int = 2,  # boundary thickness in pixels (if boundary=True)
    alpha: float = 0.6,  # overlay opacity
    gt_color: str = "lime",
    pred_color: str = "magenta",
    figsize: tuple = (12, 6),
    titles: tuple = ("Image + GT overlay", "Image + Pred overlay"),
    save_path: str = None,
    iou: float = None,
    f1: float = None,
):
    """
    Show image with GT overlay (left) and prediction overlay (right).

    Args:
        img: (H,W) grayscale or (H,W,3|4) image. Any dtype; auto-normalized to [0,1].
        gt_masks, pred_masks: labeled mask (H,W) with background=0 or stack (N,H,W) bool.
            if not given, simply save image+pred overlay singled out.
        boundary: if True, draw boundaries; else fill the union of masks.
        thickness: boundary dilation (pixels) if boundary=True.
        alpha: overlay opacity.
        gt_color, pred_color: Matplotlib color strings.
        figsize, titles: figure size and per-panel titles.
        iou: mean IoU to display (optional)
        f1: F1@0.5 to display (optional)

    Returns:
        (fig, axes) for further tweaking/saving.
    """

    def _to_rgb01(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr[..., :3].astype(np.float32)
        # normalize to [0,1]
        amin, amax = float(arr.min()), float(arr.max())
        if amax > 1.0 or amin < 0.0:
            rng = max(amax - amin, 1e-6)
            arr = (arr - amin) / rng
        return arr

    def _overlay(base: np.ndarray, masks: np.ndarray, color: str) -> np.ndarray:
        stack = _as_instance_stack(masks)  # uses your helper
        if stack.size == 0:
            return base.copy()

        if boundary:
            acc = np.zeros(base.shape[:2], dtype=bool)
            for m in stack:
                b = find_boundaries(m, mode="inner")
                if thickness > 1:
                    b = binary_dilation(b, iterations=thickness - 1)
                acc |= b
        else:
            acc = np.any(stack, axis=0)

        out = base.copy()
        col = np.array(mpl.colors.to_rgb(color), dtype=np.float32)
        out[acc] = (1.0 - alpha) * out[acc] + alpha * col
        return out

    base = _to_rgb01(img)
    if gt_masks is None:
        # Simply save image+pred
        im = Image.fromarray(_overlay(base, pred_masks, pred_color).astype(np.uint8))
        im.save(save_path)
        return
    else:
        left = _overlay(base, gt_masks, gt_color)
        right = _overlay(base, pred_masks, pred_color)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(left)
    axes[0].set_title(titles[0])
    axes[0].axis("off")
    # Compose right title with metrics if provided
    if iou is not None or f1 is not None:
        metric_str = []
        if iou is not None:
            metric_str.append(f"IoU: {iou:.3f}")
        if f1 is not None:
            metric_str.append(f"F1@0.5: {f1:.3f}")
        right_title = f"{titles[1]}\n" + ", ".join(metric_str)
    else:
        right_title = titles[1]
    axes[1].imshow(right)
    axes[1].set_title(right_title)
    axes[1].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return fig, axes
