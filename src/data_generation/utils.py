import numpy as np

from data_generation.config import SyntheticDataConfig
from data_generation.sawtooth_profile import create_sawtooth_profile


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
                max_length=np.random.uniform(cfg.min_length + 5, cfg.max_length),
                min_length=np.random.uniform(cfg.min_length, cfg.min_length + 10),
                grow_freq=cfg.grow_freq,
                shrink_freq=cfg.shrink_freq,
                noise_std=cfg.profile_noise,
                offset=np.random.randint(0, cfg.num_frames),
                fps=cfg.fps,
            ),
        )
        for _ in range(cfg.num_tubulus)
    ]


def add_gaussian(image, pos, sigma_x, sigma_y):
    if sigma_x > 0 and sigma_y > 0:
        x = np.arange(0, image.shape[1], 1)
        y = np.arange(0, image.shape[0], 1)
        x, y = np.meshgrid(x, y)
        gaussian = np.exp(-(((x - pos[0]) ** 2) / (2 * sigma_x ** 2) +
                            ((y - pos[1]) ** 2) / (2 * sigma_y ** 2)))
        image += gaussian
    return image


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
    end_x = np.clip(end_x, margin, img_size[1] - margin)
    end_y = np.clip(end_y, margin, img_size[0] - margin)

    return np.array([end_x, end_y])
