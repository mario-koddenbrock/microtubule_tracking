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
    tubulus_seeds = get_poisson_seeds(
        img_size=cfg.img_size,
        margin=cfg.margin,
        min_dist=cfg.tubuli_min_dist,
        max_tubuli=cfg.num_tubulus
    )

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
    usable_width = img_size[1] - 2 * margin
    usable_height = img_size[0] - 2 * margin

    # Bridson's algorithm for better Poisson disk sampling
    # Step 1: Define the grid for acceleration
    cell_size = min_dist / np.sqrt(2)
    grid_width = int(np.ceil(usable_width / cell_size))
    grid_height = int(np.ceil(usable_height / cell_size))
    grid = [None] * (grid_width * grid_height)

    # Maps point to grid cell
    def get_cell_idx(point):
        x, y = point
        grid_x = int((x - margin) / cell_size)
        grid_y = int((y - margin) / cell_size)
        grid_x = min(max(0, grid_x), grid_width - 1)
        grid_y = min(max(0, grid_y), grid_height - 1)
        return grid_y * grid_width + grid_x

    # Check if point is far enough from existing points
    def is_valid_point(point, points, min_dist):
        # Get nearby cells to check
        cell_x = int((point[0] - margin) / cell_size)
        cell_y = int((point[1] - margin) / cell_size)

        # Search surrounding cells (3x3 neighborhood)
        search_radius = 2
        for i in range(max(0, cell_y - search_radius), min(grid_height, cell_y + search_radius + 1)):
            for j in range(max(0, cell_x - search_radius), min(grid_width, cell_x + search_radius + 1)):
                cell_idx = i * grid_width + j
                if grid[cell_idx] is not None:
                    if distance.euclidean(point, grid[cell_idx]) < min_dist:
                        return False
        return True

    # Generate initial point
    points = []
    active_list = []

    # Add first random point
    x = np.random.uniform(margin, margin + usable_width)
    y = np.random.uniform(margin, margin + usable_height)
    first_point = (x, y)
    points.append(first_point)
    active_list.append(first_point)
    grid[get_cell_idx(first_point)] = first_point

    # Generate other points from active list
    k = 30  # Number of attempts before rejection
    while active_list and len(points) < max_tubuli:
        # Get random point from active list
        idx = np.random.randint(0, len(active_list))
        point = active_list[idx]

        # Try to find a valid new point around this point
        found = False
        for _ in range(k):
            # Generate point at distance between r and 2r from the source
            theta = np.random.uniform(0, 2 * np.pi)
            radius = min_dist * (1 + np.random.uniform(0, 1))
            new_x = point[0] + radius * np.cos(theta)
            new_y = point[1] + radius * np.sin(theta)
            new_point = (new_x, new_y)

            # Check if within bounds and valid
            if (margin <= new_x <= margin + usable_width and
                    margin <= new_y <= margin + usable_height and
                    is_valid_point(new_point, points, min_dist)):
                points.append(new_point)
                active_list.append(new_point)
                grid[get_cell_idx(new_point)] = new_point
                found = True
                break

        # If no valid point found, remove from active list
        if not found:
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
    spots (with standard deviations sigma_x, sigma_y and given contrast) at regular
    intervals (~0.5 px) along the line. This modifies `frame` in place.

    Parameters:
        frame:      2D numpy array of shape (H, W), float32 or float64, holding the image.
        mask:       2D numpy array of shape (H, W), float32 or float64, holding the mask.
        start_pt:   length-2 array-like (x0, y0) giving the line’s starting coordinates.
        end_pt:     length-2 array-like (x1, y1) giving the line’s end coordinates.
        sigma_x:    Standard deviation of the Gaussian in the x-direction (pixels).
        sigma_y:    Standard deviation of the Gaussian in the y-direction (pixels).
        contrast:   Scalar multiplier for each Gaussian spot’s amplitude.
        mask_idx:   Index for the mask, if applicable (default 0).

    Notes:
        - If the line length is zero (start == end), this simply draws one Gaussian at start_pt.
        - For performance, we compute a meshgrid (frame_x, frame_y) once per call,
          though in a tight loop you might precompute it outside and pass it in.
    """
    H, W = frame.shape
    # Create coordinate grids for all pixel centers
    yy, xx = np.mgrid[0: H, 0: W]  # yy[i,j]=i, xx[i,j]=j

    # Convert inputs to numpy arrays of dtype float
    x0, y0 = float(start_pt[0]), float(start_pt[1])
    x1, y1 = float(end_pt[0]), float(end_pt[1])

    # Vector from start to end
    vec = np.array([x1 - x0, y1 - y0], dtype=np.float32)
    length = np.linalg.norm(vec)

    # If the segment has zero length, just draw a single Gaussian at (x0, y0)
    if length == 0:
        dx = xx - x0
        dy = yy - y0
        gaussian = np.exp(-((dx ** 2) / (2 * sigma_x ** 2) + (dy ** 2) / (2 * sigma_y ** 2)))
        frame += contrast * gaussian
        mask[yy, xx] = mask_idx  # Set mask at the single point
        return frame, mask

    # Unit direction vector
    direction = vec / length

    # Choose step size ≈0.5 px (adjust for smoother/faster drawing)
    step = 0.5
    num_steps = int(np.ceil(length / step))

    # For each sample point along the line, place a 2D Gaussian:
    for i in range(num_steps + 1):
        t = i / num_steps
        x = x0 + t * vec[0]
        y = y0 + t * vec[1]

        # Compute squared distances from (x, y) to every pixel center:
        dx = xx - x
        dy = yy - y
        gaussian = np.exp(-((dx ** 2) / (2 * sigma_x ** 2) + (dy ** 2) / (2 * sigma_y ** 2)))

        # Accumulate into the frame
        frame += contrast * gaussian

        # Update the mask at this point
        center_pt = (int(np.round(x)), int(np.round(y)))
        cv2.circle(mask, center_pt, 1, mask_idx, thickness=1, lineType=8, shift=0)

    return frame, mask