from typing import Tuple

import cv2
import numpy as np
from scipy.spatial import distance

from config.synthetic_data import SyntheticDataConfig
from .sawtooth_profile import create_sawtooth_profile
from scipy.ndimage import gaussian_filter


def build_motion_seeds(cfg: SyntheticDataConfig):
    """Pre‑compute slope/intercept pairs *and* their motion profiles using Poisson sampling."""

    # Step 1: Get evenly spaced seeds
    tubulus_seeds = get_poisson_seeds(
        img_size=cfg.img_size,
        margin=cfg.margin,
        min_dist=cfg.tubuli_min_dist,
        max_tubuli=cfg.num_tubulus
    )

    # Step 2: Attach motion profiles
    motion_seeds = []
    for (slope_intercept, center) in tubulus_seeds:
        motion_profile = create_sawtooth_profile(
            num_frames=cfg.num_frames,
            max_length=np.random.randint(cfg.max_length_min, cfg.max_length_max + 1),
            min_length=np.random.randint(cfg.min_length_min, cfg.min_length_max + 1),
            grow_frames=cfg.grow_frames,
            shrink_frames=cfg.shrink_frames,
            noise_std=cfg.profile_noise,
            offset=np.random.randint(0, cfg.num_frames),
            pause_on_min_length=np.random.randint(0, cfg.pause_on_min_length + 1),
            pause_on_max_length=np.random.randint(0, cfg.pause_on_max_length + 1),
        )
        motion_seeds.append(((slope_intercept, center), motion_profile))

    return motion_seeds


def draw_tubulus(image, center, length_std, width_std, contrast=1.0):
    """
    Draws a simulated tubulus (e.g., microtubule) on the image as an anisotropic Gaussian.

    Parameters:
    - image: 2D numpy array to draw on
    - center: (x, y) coordinates of the tubulus center
    - length_std: standard deviation along the long axis (X)
    - width_std: standard deviation along the short axis (Y)
    - contrast: peak intensity to add (relative to background)
    """
    if length_std > 0 and width_std > 0:
        x = np.arange(0, image.shape[1])
        y = np.arange(0, image.shape[0])
        x, y = np.meshgrid(x, y)
        gaussian = np.exp(-(((x - center[0]) ** 2) / (2 * length_std ** 2) +
                            ((y - center[1]) ** 2) / (2 * width_std ** 2)))
        image += contrast * gaussian
    return image



def draw_spots(img, spot_coords, intensity, radii, kernel_size, sigma):
    h, w = img.shape

    if isinstance(intensity, list):
        if len(intensity) != len(spot_coords):
            raise ValueError("Intensity list must match the length of spot coordinates.")
    else:
        intensity = [intensity] * len(spot_coords)

    for idx, (y, x) in enumerate(spot_coords):
        mask = np.zeros((h, w), dtype=np.float32)
        mask = cv2.circle(mask, (int(x), int(y)), radii[idx], intensity[idx], -1)
        kernel = 2 * kernel_size[idx] + 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), sigma)
        img += mask
    return img


def add_fixed_spots(img: np.ndarray, cfg: SyntheticDataConfig) -> np.ndarray:
    n_spots = cfg.fixed_spot_count
    kernel_size = [np.random.randint(cfg.fixed_spot_kernel_size_min, cfg.fixed_spot_kernel_size_max + 1) for _ in range(n_spots)]
    sigma = cfg.fixed_spot_sigma
    h, w = img.shape

    if not hasattr(cfg, "_fixed_spot_coords"):
        cfg._fixed_spot_coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(n_spots)]

    if not hasattr(cfg, "_fixed_spot_intensities"):
        cfg._fixed_spot_intensities = [np.random.uniform(cfg.fixed_spot_intensity_min, cfg.fixed_spot_intensity_max + 1) for _ in range(n_spots)]

    if not hasattr(cfg, "_fixed_spot_radii"):
        cfg._fixed_spot_radii = [np.random.randint(cfg.fixed_spot_radius_min, cfg.fixed_spot_radius_max + 1) for _ in range(n_spots)]

    img = draw_spots(img, cfg._fixed_spot_coords, cfg._fixed_spot_intensities, cfg._fixed_spot_radii, kernel_size, sigma)
    return img

def add_moving_spots(img: np.ndarray, cfg: SyntheticDataConfig) -> np.ndarray:
    n_spots = cfg.moving_spot_count
    intensity = [np.random.uniform(cfg.moving_spot_intensity_min, cfg.moving_spot_intensity_max) for _ in range(n_spots)]
    kernel_size = [np.random.randint(cfg.moving_spot_kernel_size_min, cfg.moving_spot_kernel_size_max + 1) for _ in range(n_spots)]
    radii = [np.random.randint(cfg.moving_spot_radius_min, cfg.moving_spot_radius_max + 1) for _ in range(n_spots)]
    sigma = cfg.moving_spot_sigma
    h, w = img.shape
    moving_spot_coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(n_spots)]
    img = draw_spots(img, moving_spot_coords, intensity, radii, kernel_size, sigma)
    return img


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


def get_poisson_seeds(img_size: tuple[int, int], margin: int, min_dist: int, max_tubuli: int = 100):
    """
    Generate tubuli seeds using Poisson disk sampling to ensure even spacing.

    Returns:
        List of tuples: [(slope, intercept), center_point]
    """
    usable_width = img_size[1] - 2 * margin
    usable_height = img_size[0] - 2 * margin

    def is_far_enough(point, points, min_dist):
        return all(distance.euclidean(point, p) >= min_dist for p in points)

    # Sample Poisson points manually (basic rejection sampling version)
    points = []
    attempts = 0
    max_attempts = max_tubuli * 50
    while len(points) < max_tubuli and attempts < max_attempts:
        x = np.random.uniform(margin, margin + usable_width)
        y = np.random.uniform(margin, margin + usable_height)
        candidate = (x, y)
        if is_far_enough(candidate, points, min_dist):
            points.append(candidate)
        attempts += 1

    seeds = []
    for x, y in points:
        slope = np.random.uniform(-1.5, 1.5)
        intercept = y - slope * x
        seeds.append(((slope, intercept), (x, y)))

    return seeds



def grow_shrink_seed(frame, original, slope, motion_profile, img_size: tuple[int, int], margin: int):
    net_motion = motion_profile[frame]

    dx = net_motion / np.sqrt(1 + slope ** 2)
    dy = slope * dx

    end_x = original[0] + dx
    end_y = original[1] + dy

    # Clip to safe margin
    # end_x = np.clip(end_x, margin, img_size[1] - margin)
    # end_y = np.clip(end_y, margin, img_size[0] - margin)

    return np.array([end_x, end_y])

def annotate_frame(frame, frame_idx, fps=5, show_time=True, show_scale=True, scale_um_per_pixel=0.1, scale_length_um=5):
    """
    Annotate the frame with optional time and scale bar.

    Parameters:
    - frame: The image (H×W×3, uint8)
    - frame_idx: Index of the current frame
    - fps: Frames per second (for computing time)
    - show_time: Whether to draw time in top-left corner
    - show_scale: Whether to draw scale bar in bottom-right corner
    - scale_um_per_pixel: micrometers per pixel
    - scale_length_um: length of the scale bar in micrometers
    """
    annotated = frame.copy()
    annotation_color = (0, 0, 0)
    H, W = annotated.shape[:2]

    # 🕒 Time annotation
    if show_time:
        time_sec = frame_idx / fps
        time_str = f"{int(time_sec):d}:{int((time_sec % 1) * 100):02d}"
        cv2.putText(annotated, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, annotation_color, 2, cv2.LINE_AA)

    # 📏 Scale bar
    if show_scale:
        scale_length_px = int(scale_length_um / scale_um_per_pixel)
        bar_height = 6  # pixels
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