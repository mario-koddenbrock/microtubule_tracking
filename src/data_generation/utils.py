# FILE: data_generation/utils.py

from typing import Tuple, List

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance

from config.synthetic_data import SyntheticDataConfig


def build_motion_seeds(cfg: SyntheticDataConfig) -> List[np.ndarray]:
    """
    Precompute the base anchor point for each microtubule. The microtubule object
    will handle its own motion profile generation.
    """
    tubulus_seeds = get_random_seeds(
        img_size=cfg.img_size,
        margin=cfg.margin,
        min_dist=cfg.tubuli_seed_min_dist,
        max_tubuli=cfg.num_tubuli,
    )
    start_points = [
        np.array(center, dtype=np.float32) for (_slope_intercept, center) in tubulus_seeds
    ]
    return start_points


def draw_tubulus(image, center, length_std, width_std, contrast=1.0):
    """
    Draws a simulated tubulus (e.g., microtubule) on the image as an anisotropic Gaussian.
    """
    if length_std > 0 and width_std > 0:
        x = np.arange(0, image.shape[1])
        y = np.arange(0, image.shape[0])
        x, y = np.meshgrid(x, y)
        gaussian = np.exp(
            -(
                ((x - center[0]) ** 2) / (2 * length_std**2)
                + ((y - center[1]) ** 2) / (2 * width_std**2)
            )
        )
        image += contrast * gaussian
    return image


def apply_global_blur(img: np.ndarray, cfg) -> np.ndarray:
    """Apply a soft blur to the entire image."""
    sigma = cfg.global_blur_sigma
    return gaussian_filter(img, sigma=sigma) if sigma > 0 else img


def get_random_seeds(img_size: tuple[int, int], margin: int, min_dist: int, max_tubuli: int = 100):
    usable_min_x = margin
    usable_max_x = img_size[1] - margin
    usable_min_y = margin
    usable_max_y = img_size[0] - margin
    if usable_max_x <= usable_min_x or usable_max_y <= usable_min_y:
        return []
    points = []
    max_attempts = max_tubuli * 100
    attempts = 0
    while len(points) < max_tubuli and attempts < max_attempts:
        candidate_x = np.random.uniform(usable_min_x, usable_max_x)
        candidate_y = np.random.uniform(usable_min_y, usable_max_y)
        candidate_point = (candidate_x, candidate_y)
        is_valid = True
        for existing_point in points:
            if distance.euclidean(candidate_point, existing_point) < min_dist:
                is_valid = False
                break
        if is_valid:
            points.append(candidate_point)
        attempts += 1
    if attempts >= max_attempts and len(points) < max_tubuli:
        print(
            f"Warning: Reached max attempts ({max_attempts}) before finding all tubuli. Generated {len(points)} out of {max_tubuli} requested."
        )
    seeds = []
    for x, y in points:
        slope = np.random.uniform(-1.5, 1.5)
        intercept = y - slope * x
        seeds.append(((slope, intercept), (x, y)))
    return seeds


def compute_vignette(cfg: SyntheticDataConfig) -> np.ndarray:
    if cfg.vignetting_strength <= 0.0:
        return 1.0
    yy, xx = np.mgrid[: cfg.img_size[0], : cfg.img_size[1]]
    norm_x = (xx - cfg.img_size[1] / 2) / (cfg.img_size[1] / 2)
    norm_y = (yy - cfg.img_size[0] / 2) / (cfg.img_size[0] / 2)
    vignette = 1.0 - cfg.vignetting_strength * (norm_x**2 + norm_y**2)
    return np.clip(vignette, 0.5, 1.0)


def draw_gaussian_line_on_rgb(
    frame: np.ndarray,
    mask: np.ndarray,
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    color_contrast_rgb: Tuple[float, float, float],
    mask_idx: int,
    additional_mask: np.ndarray | None = None,
):
    """
    Rasterizes a line by placing small 2D Gaussian spots at regular intervals.

    This function is refactored for clarity by defining a nested helper,
    `draw_spot_at_point`, to avoid duplicating the core drawing logic.
    """
    H, W, C = frame.shape
    assert C == 3, "Frame must be a 3-channel RGB image."

    # Pre-compute coordinate grids and line properties
    yy, xx = np.mgrid[0:H, 0:W]
    x0, y0 = float(start_pt[0]), float(start_pt[1])
    x1, y1 = float(end_pt[0]), float(end_pt[1])
    vec = np.array([x1 - x0, y1 - y0], dtype=np.float32)
    length = np.linalg.norm(vec)
    mask_threshold = 0.01

    # --- Nested Helper Function for Drawing a Single Spot ---
    # This encapsulates the core logic, which prevents code duplication.
    # It modifies the `frame` and `mask` arrays from the outer scope in-place.
    def draw_spot_at_point(px: float, py: float):
        """Calculates and applies a single Gaussian spot centered at (px, py)."""
        # Calculate distances of all grid points from the spot's center
        dx = xx - px
        dy = yy - py

        # Calculate the 2D Gaussian blob
        gaussian_blob = np.exp(-((dx**2) / (2 * sigma_x**2) + (dy**2) / (2 * sigma_y**2)))

        # Apply the contrast to each RGB channel and add it to the frame
        for c in range(3):
            frame[..., c] += color_contrast_rgb[c] * gaussian_blob

        # Update the instance mask where the Gaussian is strong enough
        if mask is not None:
            mask[gaussian_blob > mask_threshold] = mask_idx
        if additional_mask is not None:
            additional_mask[gaussian_blob > mask_threshold] = mask_idx

    # --- End of Helper Function ---

    # If the line has no length, just draw a single spot at the start point.
    if length < 1e-6:
        draw_spot_at_point(x0, y0)
        return

    # For lines with length, iterate along the line and draw spots at each step.
    step = 0.5  # Draw a spot every half a pixel for smooth coverage
    num_steps = int(np.ceil(length / step))
    if num_steps == 0:
        num_steps = 1

    for i in range(num_steps + 1):
        t = i / num_steps
        px = x0 + t * vec[0]
        py = y0 + t * vec[1]
        draw_spot_at_point(px, py)


def annotate_frame(frame, cfg, frame_idx):
    """Annotates the frame using the color from the config."""
    annotated = frame.copy()
    # Convert 0-1 float color to 0-255 uint8 BGR tuple for OpenCV.
    color_bgr = tuple(int(c * 255) for c in reversed(cfg.annotation_color_rgb))

    H, W = annotated.shape[:2]

    # Time annotation
    if cfg.show_time:
        time_sec = frame_idx / cfg.fps
        time_str = f"{int(time_sec):d}:{int((time_sec % 1) * 100):02d}"
        cv2.putText(
            annotated, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 2, cv2.LINE_AA
        )

    # Scale bar
    if cfg.show_scale:
        scale_length_px = int(cfg.scale_bar_um / cfg.um_per_pixel)
        bar_height = 6
        x_end, y_start = W - 10, H - 20
        x_start, y_end = x_end - scale_length_px, y_start - bar_height
        cv2.rectangle(annotated, (x_start, y_end), (x_end, y_start), color_bgr, -1)

    return annotated
