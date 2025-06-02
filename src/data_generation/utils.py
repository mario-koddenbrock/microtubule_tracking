import cv2
import numpy as np

from data_generation.config import SyntheticDataConfig
from data_generation.sawtooth_profile import create_sawtooth_profile
from scipy.ndimage import gaussian_filter

def build_motion_seeds(cfg: SyntheticDataConfig):
    """Pre‑compute slope/intercept pairs *and* their motion profiles.

    Keeping the RNG separate from the rendering loop makes the whole pipeline
    deterministic and lets us reproduce exact sequences from a single call.
    """
    return [
        (
            get_seed(cfg.img_size, cfg.margin),
            create_sawtooth_profile(
                num_frames=cfg.num_frames,
                max_length=np.random.randint(cfg.max_length_min, cfg.max_length_max + 1),
                min_length=np.random.randint(cfg.min_length_min, cfg.min_length_max + 1),
                grow_frames=cfg.grow_frames,
                shrink_frames=cfg.shrink_frames,
                noise_std=cfg.profile_noise,
                offset=np.random.randint(0, cfg.num_frames),
                pause_on_min_length=np.random.randint(0, cfg.pause_on_min_length + 1),
                pause_on_max_length=np.random.randint(0, cfg.pause_on_max_length + 1),
            ),
        )
        for _ in range(cfg.num_tubulus)
    ]


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


def add_fixed_spots(img: np.ndarray, cfg: SyntheticDataConfig, rng: np.random.Generator) -> np.ndarray:
    n_spots = cfg.fixed_spot_count
    intensity = cfg.fixed_spot_contrast
    kernel_size = cfg.fixed_spot_kernel_size
    sigma = cfg.fixed_spot_sigma
    h, w = img.shape

    if not hasattr(cfg, "_fixed_spot_coords"):
        cfg._fixed_spot_coords = [(rng.integers(0, h), rng.integers(0, w)) for _ in range(n_spots)]

    img = draw_spots(img, cfg._fixed_spot_coords, intensity, kernel_size, sigma)
    return img


def draw_spots(img, spot_coords, intensity, kernel_size, sigma):
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.float32)
    for y, x in spot_coords:
        mask = cv2.circle(mask, (int(x), int(y)), 1, intensity, -1)
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
    # #plot for debugging
    # import matplotlib.pyplot as plt
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    img += mask
    return img


def add_moving_spots(img: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    n_spots = cfg.moving_spot_count
    intensity = cfg.moving_spot_contrast
    kernel_size = cfg.moving_spot_kernel_size
    sigma = cfg.moving_spot_sigma
    h, w = img.shape
    moving_spot_coords = [(rng.integers(0, h), rng.integers(0, w)) for _ in range(n_spots)]
    img = draw_spots(img, moving_spot_coords, intensity, kernel_size, sigma)
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


def get_seed(img_size: tuple[int, int], margin: int):
    usable_width = img_size[1] - 2 * margin
    usable_height = img_size[0] - 2 * margin
    start_x = np.random.uniform(margin, margin + usable_width)
    start_y = np.random.uniform(margin, margin + usable_height)
    slope = np.random.uniform(-1.5, 1.5)
    intercept = start_y - slope * start_x

    return np.array([slope, intercept]), np.array([start_x, start_y])


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
