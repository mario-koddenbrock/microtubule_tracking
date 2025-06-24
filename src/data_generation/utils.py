from typing import Tuple, List

import cv2
import numpy as np
from scipy.spatial import distance

from config.spots import SpotConfig
from config.synthetic_data import SyntheticDataConfig
from .sawtooth_profile import create_sawtooth_profile
from scipy.ndimage import gaussian_filter


def build_motion_seeds(
    cfg: SyntheticDataConfig
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Precompute, for each microtubule, its base anchor point (center) and a
    length‐over‐time (motion_profile) array. We ignore 'slope_intercept' entirely.

    Returns a list of tuples:
        [(start_pt_1, motion_profile_1), (start_pt_2, motion_profile_2), …]
    """

    # Step 1: Get random seed positions via Poisson‐disk sampling.
    #    get_poisson_seeds returns a list of ((slope, intercept), center) pairs.
    tubulus_seeds = get_random_seeds(
        img_size=cfg.img_size,
        margin=cfg.margin,
        min_dist=cfg.tubuli_min_dist,
        max_tubuli=cfg.num_tubulus
    )

    # x_coords = [p[1][0] for p in tubulus_seeds]
    # y_coords = [p[1][1] for p in tubulus_seeds]
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.scatter(x_coords, y_coords)
    # ax.set_xlim(0, cfg.img_size[1])
    # ax.set_ylim(0, cfg.img_size[0])
    # ax.set_aspect('equal', adjustable='box')
    # ax.invert_yaxis()  # Match image coordinate system (0,0 at top-left)
    # plt.title(f"Generated {len(tubulus_seeds)} Points")
    # plt.show()

    motion_seeds: List[Tuple[np.ndarray, np.ndarray]] = []
    for (_slope_intercept, center) in tubulus_seeds:
        # We no longer care about slope_intercept, so just ignore it.
        start_pt = np.array(center, dtype=np.float32)

        # Build a sawtooth (or sinusoid) length profile:
        max_len = np.random.randint(cfg.max_length_min, cfg.max_length_max + 1)
        min_len = np.random.randint(cfg.min_length_min, cfg.min_length_max + 1)
        motion_profile = create_sawtooth_profile(
            num_frames=cfg.num_frames,
            max_length=max_len,
            min_length=min_len,
            grow_frames=cfg.grow_frames,
            shrink_frames=cfg.shrink_frames,
            noise_std=cfg.profile_noise,
            offset=np.random.randint(0, cfg.num_frames),
            pause_on_min_length=np.random.randint(0, cfg.pause_on_min_length + 1),
            pause_on_max_length=np.random.randint(0, cfg.pause_on_max_length + 1),
        )

        motion_seeds.append((start_pt, motion_profile))

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


def apply_random_spots(img: np.ndarray, spot_cfg: SpotConfig) -> np.ndarray:
    """
    Adds stateless spots that are regenerated completely on every frame.

    Args:
        img: The image to draw on.
        spot_cfg: The configuration for the random spots.

    Returns:
        The image with random spots added.
    """
    n_spots = spot_cfg.count
    if n_spots == 0:
        return img

    h, w = img.shape

    # Generate all properties on-the-fly for each frame
    coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(n_spots)]
    intensities = [np.random.uniform(spot_cfg.intensity_min, spot_cfg.intensity_max) for _ in range(n_spots)]
    radii = [np.random.randint(spot_cfg.radius_min, spot_cfg.radius_max + 1) for _ in range(n_spots)]
    kernel_sizes = [np.random.randint(spot_cfg.kernel_size_min, spot_cfg.kernel_size_max + 1) for _ in range(n_spots)]

    return draw_spots(img, coords, intensities, radii, kernel_sizes, spot_cfg.sigma)


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
    """
    Generates tubuli seeds using simple random rejection sampling.

    This method is simpler than Poisson-disk sampling but can be much slower if
    the number of tubuli is high or the minimum distance is large, as it becomes
    harder to find valid empty spots.

    Args:
        img_size: (height, width) of the image.
        margin: Margin from the edges where no points will be placed.
        min_dist: The minimum required distance between the center of any two tubuli.
        max_tubuli: The maximum number of tubuli to generate.

    Returns:
        List of tuples: [(slope, intercept), center_point]
    """
    # Define the usable area for placing points
    usable_min_x = margin
    usable_max_x = img_size[1] - margin
    usable_min_y = margin
    usable_max_y = img_size[0] - margin

    if usable_max_x <= usable_min_x or usable_max_y <= usable_min_y:
        print("Warning: Usable area is zero or negative due to large margin. No points will be generated.")
        return []

    points = []

    # To prevent an infinite loop if the area becomes too crowded to place new points,
    # we'll set a limit on the number of failed attempts.
    # A reasonable limit is 100 failed attempts for every point we want to place.
    max_attempts = max_tubuli * 100
    attempts = 0

    while len(points) < max_tubuli and attempts < max_attempts:
        # Step 1: Generate a random candidate point within the usable area
        candidate_x = np.random.uniform(usable_min_x, usable_max_x)
        candidate_y = np.random.uniform(usable_min_y, usable_max_y)
        candidate_point = (candidate_x, candidate_y)

        # Step 2: Check if the candidate is too close to any existing point
        is_valid = True
        for existing_point in points:
            if distance.euclidean(candidate_point, existing_point) < min_dist:
                is_valid = False
                break  # No need to check other points, it's already invalid

        # Step 3: If it's valid, add it to our list. Otherwise, do nothing.
        if is_valid:
            points.append(candidate_point)

        attempts += 1

    # If the loop terminated due to reaching max attempts, print a warning
    if attempts >= max_attempts and len(points) < max_tubuli:
        print(f"Warning: Reached max attempts ({max_attempts}) before finding all tubuli. "
              f"Generated {len(points)} out of {max_tubuli} requested. "
              "Consider reducing min_dist or max_tubuli.")

    # Convert the final points into the desired seed format with slope and intercept
    seeds = []
    for x, y in points:
        slope = np.random.uniform(-1.5, 1.5)
        intercept = y - slope * x
        seeds.append(((slope, intercept), (x, y)))

    return seeds


def get_poisson_seeds(img_size: tuple[int, int], margin: int, min_dist: int, max_tubuli: int = 100):
    """
    Generate tubuli seeds using improved Poisson disk sampling to ensure even spacing.

    Args:
        img_size: (height, width) of the image
        margin: margin from the edges
        min_dist: minimum distance between seeds
        max_tubuli: maximum number of tubuli to generate

    Returns:
        List of tuples: [(slope, intercept), center_point]
    """
    if min_dist <= 0:
        raise ValueError("min_dist must be positive.")

    usable_width = img_size[1] - 2 * margin
    usable_height = img_size[0] - 2 * margin

    if usable_width <= 0 or usable_height <= 0:
        print("Warning: Usable area is zero or negative due to large margin. No points will be generated.")
        return []

    # Step 1: Define the grid for acceleration
    cell_size = min_dist / np.sqrt(2)
    grid_width = int(np.ceil(usable_width / cell_size))
    grid_height = int(np.ceil(usable_height / cell_size))
    grid = [None] * (grid_width * grid_height)

    def get_grid_coords(point):
        x, y = point
        # This translates image coordinates to grid cell coordinates (0-indexed)
        grid_x = int((x - margin) / cell_size)
        grid_y = int((y - margin) / cell_size)
        return grid_x, grid_y

    def is_valid_point(point):
        grid_x, grid_y = get_grid_coords(point)

        # Search the surrounding 5x5 neighborhood of cells for collisions
        search_radius = 2
        for i in range(max(0, grid_y - search_radius), min(grid_height, grid_y + search_radius + 1)):
            for j in range(max(0, grid_x - search_radius), min(grid_width, grid_x + search_radius + 1)):
                cell_idx = i * grid_width + j
                existing_point = grid[cell_idx]
                if existing_point and distance.euclidean(point, existing_point) < min_dist:
                    return False
        return True

    # ---- Main Algorithm ----
    points = []
    active_list = []

    # Add the first random point
    x = margin + np.random.uniform(0, usable_width)
    y = margin + np.random.uniform(0, usable_height)
    first_point = (x, y)

    active_list.append(first_point)
    points.append(first_point)
    grid_x, grid_y = get_grid_coords(first_point)
    grid[grid_y * grid_width + grid_x] = first_point

    k = 30  # Number of attempts before deactivating a point
    while active_list and len(points) < max_tubuli:
        idx = np.random.randint(0, len(active_list))
        source_point = active_list[idx]

        found_new_point = False
        for _ in range(k):
            theta = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(min_dist, 2 * min_dist)
            new_x = source_point[0] + radius * np.cos(theta)
            new_y = source_point[1] + radius * np.sin(theta)
            new_point = (new_x, new_y)

            # CRUCIAL: Check bounds FIRST, then validate distance.
            # Use strict inequality for the upper bound to avoid the boundary bug.
            if (margin <= new_x < margin + usable_width and
                    margin <= new_y < margin + usable_height and
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


def draw_gaussian_line(frame: np.ndarray,
                       mask: np.ndarray,
                       start_pt: np.ndarray,
                       end_pt: np.ndarray,
                       sigma_x: float,
                       sigma_y: float,
                       contrast: float,
                       mask_idx: int = 0,
                       ):
    """
    Rasterize a straight line from start_pt → end_pt by placing small 2D Gaussian
    spots at regular intervals. Modifies `frame` and `mask` in place.

    This version uses safe NumPy indexing to modify the mask, avoiding memory
    corruption associated with using cv2.circle on uint16 arrays.
    """
    H, W = frame.shape
    # Create coordinate grids for all pixel centers.
    # We only need to do this once.
    yy, xx = np.mgrid[0:H, 0:W]

    x0, y0 = float(start_pt[0]), float(start_pt[1])
    x1, y1 = float(end_pt[0]), float(end_pt[1])

    vec = np.array([x1 - x0, y1 - y0], dtype=np.float32)
    length = np.linalg.norm(vec)

    # Define a threshold to determine which pixels belong to the object in the mask
    # A small fraction of the peak contrast is a good choice.
    mask_threshold = 0.01

    # If the segment has zero length, just draw a single Gaussian
    if length < 1e-6:  # Use a small tolerance for floating point comparison
        dx = xx - x0
        dy = yy - y0
        gaussian = np.exp(-((dx ** 2) / (2 * sigma_x ** 2) + (dy ** 2) / (2 * sigma_y ** 2)))

        # Add the gaussian to the frame
        frame += contrast * gaussian

        # FIXED: Use safe NumPy boolean indexing to set the mask.
        # This only sets the mask where the gaussian is strong enough.
        mask[gaussian > mask_threshold] = mask_idx
        return  # No need to return frame/mask as they are modified in-place

    # For lines longer than 0, iterate along the line
    step = 0.5
    num_steps = int(np.ceil(length / step))
    # Avoid division by zero if num_steps is 0 for very short lines
    if num_steps == 0:
        num_steps = 1

    for i in range(num_steps + 1):
        t = i / num_steps
        px = x0 + t * vec[0]
        py = y0 + t * vec[1]

        # Compute gaussian centered at the current point (px, py)
        dx = xx - px
        dy = yy - py
        gaussian = np.exp(-((dx ** 2) / (2 * sigma_x ** 2) + (dy ** 2) / (2 * sigma_y ** 2)))

        # Add the gaussian shape to the main image
        frame += contrast * gaussian

        # FIXED: Use safe NumPy boolean indexing instead of cv2.circle
        # This correctly updates the uint16 mask without memory errors.
        mask[gaussian > mask_threshold] = mask_idx