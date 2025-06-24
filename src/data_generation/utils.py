# FILE: data_generation/utils.py

from typing import Tuple, List, Optional

import cv2
import numpy as np
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

from config.spots import SpotConfig
from config.synthetic_data import SyntheticDataConfig
from .spots import SpotGenerator


def build_motion_seeds(
    cfg: SyntheticDataConfig
) -> List[np.ndarray]:
    """
    Precompute the base anchor point for each microtubule. The microtubule object
    will handle its own motion profile generation.
    """
    tubulus_seeds = get_random_seeds(
        img_size=cfg.img_size,
        margin=cfg.margin,
        min_dist=cfg.tubuli_min_dist,
        max_tubuli=cfg.num_tubulus
    )
    start_points = [np.array(center, dtype=np.float32) for (_slope_intercept, center) in tubulus_seeds]
    return start_points


def draw_tubulus(image, center, length_std, width_std, contrast=1.0):
    """
    Draws a simulated tubulus (e.g., microtubule) on the image as an anisotropic Gaussian.
    """
    if length_std > 0 and width_std > 0:
        x = np.arange(0, image.shape[1])
        y = np.arange(0, image.shape[0])
        x, y = np.meshgrid(x, y)
        gaussian = np.exp(-(((x - center[0]) ** 2) / (2 * length_std ** 2) +
                            ((y - center[1]) ** 2) / (2 * width_std ** 2)))
        image += contrast * gaussian
    return image


# THIS IS THE CORRECTED FUNCTION
def apply_random_spots(img: np.ndarray, spot_cfg: SpotConfig) -> np.ndarray:
    """
    Adds stateless spots that are regenerated completely on every frame.
    This function is now RGB-aware.
    """
    n_spots = spot_cfg.count
    if n_spots == 0:
        return img

    # CHANGED: Correctly get image dimensions, handling both grayscale and RGB.
    if img.ndim == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape

    # Generate all properties on-the-fly for each frame
    coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(n_spots)]
    intensities = [np.random.uniform(spot_cfg.intensity_min, spot_cfg.intensity_max) for _ in range(n_spots)]
    radii = [np.random.randint(spot_cfg.radius_min, spot_cfg.radius_max + 1) for _ in range(n_spots)]
    kernel_sizes = [np.random.randint(spot_cfg.kernel_size_min, spot_cfg.kernel_size_max + 1) for _ in range(n_spots)]

    # This call is now safe because both this function and the called function are RGB-aware.
    return SpotGenerator.draw_spots(img, coords, intensities, radii, kernel_sizes, spot_cfg.sigma)


def apply_global_blur(img: np.ndarray, cfg) -> np.ndarray:
    """Apply a soft blur to the entire image."""
    sigma = cfg.global_blur_sigma
    return gaussian_filter(img, sigma=sigma) if sigma > 0 else img


def normalize_image(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min + 1e-8)


def poisson_noise(image, snr):
    max_val = np.max(image)
    noisy = np.random.poisson(image * snr) / snr
    return np.clip(noisy / max_val if max_val > 0 else image, 0, 1)


def get_random_seeds(img_size: tuple[int, int],
                               margin: int,
                               min_dist: int,
                               max_tubuli: int = 100):
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
        print(f"Warning: Reached max attempts ({max_attempts}) before finding all tubuli. Generated {len(points)} out of {max_tubuli} requested.")
    seeds = []
    for x, y in points:
        slope = np.random.uniform(-1.5, 1.5)
        intercept = y - slope * x
        seeds.append(((slope, intercept), (x, y)))
    return seeds


def get_poisson_seeds(img_size: tuple[int, int], margin: int, min_dist: int, max_tubuli: int = 100):
    if min_dist <= 0:
        raise ValueError("min_dist must be positive.")
    usable_width, usable_height = img_size[1] - 2 * margin, img_size[0] - 2 * margin
    if usable_width <= 0 or usable_height <= 0:
        return []
    cell_size = min_dist / np.sqrt(2)
    grid_width, grid_height = int(np.ceil(usable_width / cell_size)), int(np.ceil(usable_height / cell_size))
    grid = [None] * (grid_width * grid_height)
    def get_grid_coords(point):
        return int((point[0] - margin) / cell_size), int((point[1] - margin) / cell_size)
    def is_valid_point(point):
        grid_x, grid_y = get_grid_coords(point)
        for i in range(max(0, grid_y - 2), min(grid_height, grid_y + 3)):
            for j in range(max(0, grid_x - 2), min(grid_width, grid_x + 3)):
                cell_idx = i * grid_width + j
                existing_point = grid[cell_idx]
                if existing_point and distance.euclidean(point, existing_point) < min_dist:
                    return False
        return True
    points, active_list = [], []
    first_point = (margin + np.random.uniform(0, usable_width), margin + np.random.uniform(0, usable_height))
    active_list.append(first_point)
    points.append(first_point)
    grid_x, grid_y = get_grid_coords(first_point)
    grid[grid_y * grid_width + grid_x] = first_point
    k = 30
    while active_list and len(points) < max_tubuli:
        idx = np.random.randint(0, len(active_list))
        source_point = active_list[idx]
        found_new_point = False
        for _ in range(k):
            theta, radius = np.random.uniform(0, 2 * np.pi), np.random.uniform(min_dist, 2 * min_dist)
            new_point = (source_point[0] + radius * np.cos(theta), source_point[1] + radius * np.sin(theta))
            if (margin <= new_point[0] < margin + usable_width and
                    margin <= new_point[1] < margin + usable_height and
                    is_valid_point(new_point)):
                points.append(new_point)
                active_list.append(new_point)
                new_grid_x, new_grid_y = get_grid_coords(new_point)
                grid[new_grid_y * grid_width + new_grid_x] = new_point
                found_new_point = True
                break
        if not found_new_point:
            active_list.pop(idx)
    seeds = []
    for x, y in points:
        slope, intercept = np.random.uniform(-1.5, 1.5), y - slope * x
        seeds.append(((slope, intercept), (x, y)))
    return seeds


def grow_shrink_seed(frame, original, slope, motion_profile, img_size: tuple[int, int], margin: int):
    # ... (this function is likely unused now but left for posterity)
    net_motion = motion_profile[frame]
    dx = net_motion / np.sqrt(1 + slope ** 2)
    dy = slope * dx
    end_x = original[0] + dx
    end_y = original[1] + dy
    return np.array([end_x, end_y])


def annotate_frame(frame, frame_idx, fps=5, show_time=True, show_scale=True, scale_um_per_pixel=0.1, scale_length_um=5):
    annotated = frame.copy()
    annotation_color = (0, 0, 0)
    H, W = annotated.shape[:2]
    if show_time:
        time_sec = frame_idx / fps
        time_str = f"{int(time_sec):d}:{int((time_sec % 1) * 100):02d}"
        cv2.putText(annotated, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, annotation_color, 2, cv2.LINE_AA)
    if show_scale:
        scale_length_px = int(scale_length_um / scale_um_per_pixel)
        bar_height = 6
        x_end = W - 10
        x_start = x_end - scale_length_px
        y_start = H - 20
        y_end = y_start - bar_height
        cv2.rectangle(annotated, (x_start, y_end), (x_end, y_start), annotation_color, -1)
    return annotated


def compute_vignette(cfg: SyntheticDataConfig) -> np.ndarray:
    if cfg.vignetting_strength <= 0.0:
        return 1.0
    yy, xx = np.mgrid[:cfg.img_size[0], :cfg.img_size[1]]
    norm_x = (xx - cfg.img_size[1] / 2) / (cfg.img_size[1] / 2)
    norm_y = (yy - cfg.img_size[0] / 2) / (cfg.img_size[0] / 2)
    vignette = 1.0 - cfg.vignetting_strength * (norm_x ** 2 + norm_y ** 2)
    return np.clip(vignette, 0.5, 1.0)


def update_bend_params(cfg: SyntheticDataConfig, inst_id: int, motion_profile: np.ndarray,
                        start_pt: np.ndarray, end_pt: np.ndarray, rng: np.random.Generator) -> Tuple[float, float, bool]:
    # ... (this function is likely unused but left for posterity)
    total_length = np.linalg.norm(end_pt - start_pt)
    if not hasattr(cfg, "_bend_params"):
        cfg._bend_params = {}
    if inst_id not in cfg._bend_params:
        apply_bend = rng.random() < cfg.bend_prob
        this_amp = cfg.bend_amplitude if apply_bend else 0.0
        min_length = np.min(motion_profile)
        dynamic_straight_fraction = min(min_length / total_length, 1.0) if total_length > 0 else 1.0
        cfg._bend_params[inst_id] = (this_amp, dynamic_straight_fraction)
    else:
        this_amp, dynamic_straight_fraction = cfg._bend_params[inst_id]
        apply_bend = this_amp > 0.0
    return this_amp, dynamic_straight_fraction, apply_bend


def draw_colored_gaussian_line(frame: np.ndarray,
                               mask: np.ndarray,
                               start_pt: np.ndarray,
                               end_pt: np.ndarray,
                               sigma_x: float,
                               sigma_y: float,
                               contrast: float,
                               color_rgb: Tuple[float, float, float],
                               mask_idx: int = 0):

    H, W, C = frame.shape
    assert C == 3, "Frame must be a 3-channel RGB image."
    yy, xx = np.mgrid[0:H, 0:W]
    x0, y0 = float(start_pt[0]), float(start_pt[1])
    x1, y1 = float(end_pt[0]), float(end_pt[1])
    vec = np.array([x1 - x0, y1 - y0], dtype=np.float32)
    length = np.linalg.norm(vec)
    mask_threshold = 0.01
    if length < 1e-6:
        dx, dy = xx - x0, yy - y0
        gaussian = np.exp(-((dx ** 2) / (2 * sigma_x ** 2) + (dy ** 2) / (2 * sigma_y ** 2)))
        for i in range(3):
            frame[..., i] += contrast * color_rgb[i] * gaussian
        mask[gaussian > mask_threshold] = mask_idx
        return
    step = 0.5
    num_steps = int(np.ceil(length / step))
    if num_steps == 0: num_steps = 1
    for i in range(num_steps + 1):
        t = i / num_steps
        px, py = x0 + t * vec[0], y0 + t * vec[1]
        dx, dy = xx - px, yy - py
        gaussian = np.exp(-((dx ** 2) / (2 * sigma_x ** 2) + (dy ** 2) / (2 * sigma_y ** 2)))
        for c in range(3):
            frame[..., c] += contrast * color_rgb[c] * gaussian
        mask[gaussian > mask_threshold] = mask_idx