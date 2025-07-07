import logging
from typing import Tuple, List, Union, Optional

import albumentations as A
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance

from config.album_config import AlbumentationsConfig
from config.synthetic_data import SyntheticDataConfig

# Get a logger instance for this module
logger = logging.getLogger(f"mt.{__name__}")


def build_motion_seeds(cfg: SyntheticDataConfig) -> List[np.ndarray]:
    """
    Precompute the base anchor point for each microtubule. The microtubule object
    will handle its own motion profile generation.
    """
    logger.info("Building motion seeds for microtubules.")
    tubulus_seeds = get_random_seeds(
        img_size=cfg.img_size,
        margin=cfg.margin,
        min_dist=cfg.tubuli_seed_min_dist,
        max_tubuli=cfg.num_tubuli,
    )
    start_points = [
        np.array(center, dtype=np.float32) for (_slope_intercept, center) in tubulus_seeds
    ]
    logger.info(f"Generated {len(start_points)} motion seeds.")
    logger.debug(f"Sample start points: {start_points[:min(5, len(start_points))]}.")
    return start_points


def draw_tubulus(image: np.ndarray, center: Tuple[float, float], length_std: float, width_std: float,
                 contrast: float = 1.0) -> np.ndarray:
    """
    Draws a simulated tubulus (e.g., microtubule) on the image as an anisotropic Gaussian.
    NOTE: This function appears to be a simplified/deprecated version, `draw_gaussian_line_rgb` is typically used for actual lines.
    """
    logger.debug(
        f"Drawing single tubulus at {center} with length_std={length_std:.2f}, width_std={width_std:.2f}, contrast={contrast:.2f}.")
    if length_std <= 0 or width_std <= 0:
        logger.warning(f"Skipping draw_tubulus: length_std ({length_std}) or width_std ({width_std}) is non-positive.")
        return image

    try:
        x_grid = np.arange(0, image.shape[1])
        y_grid = np.arange(0, image.shape[0])
        x, y = np.meshgrid(x_grid, y_grid)
        gaussian = np.exp(-(((x - center[0]) ** 2) / (2 * length_std ** 2) +
                            ((y - center[1]) ** 2) / (2 * width_std ** 2)))
        image += contrast * gaussian
        logger.debug("Tubulus drawn successfully.")
    except Exception as e:
        logger.error(f"Error drawing tubulus at {center}: {e}", exc_info=True)
    return image


def apply_global_blur(img: np.ndarray, cfg: SyntheticDataConfig) -> np.ndarray:
    """Apply a soft blur to the entire image."""
    sigma = cfg.global_blur_sigma
    if sigma > 0:
        logger.debug(f"Applying global Gaussian blur with sigma={sigma:.2f}.")
        try:
            blurred_img = gaussian_filter(img, sigma=sigma)
            logger.debug("Global blur applied.")
            return blurred_img
        except Exception as e:
            logger.error(f"Error applying global blur with sigma {sigma}: {e}", exc_info=True)
            return img  # Return original image on error
    else:
        logger.debug("Skipping global blur (sigma is 0 or negative).")
        return img


def get_random_seeds(
        img_size: Tuple[int, int],
        margin: int,
        min_dist: int,
        max_tubuli: int = 100,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Generates random seed points within image boundaries, ensuring minimum distance between them.

    Returns:
        List[Tuple[Tuple[float, float], Tuple[float, float]]]: List of tuples, each containing
        ((slope, intercept), (x, y)) for the seed.
    """
    logger.debug(
        f"Generating random seeds: img_size={img_size}, margin={margin}, min_dist={min_dist}, max_tubuli={max_tubuli}.")

    usable_min_x = margin
    usable_max_x = img_size[1] - margin
    usable_min_y = margin
    usable_max_y = img_size[0] - margin

    if usable_max_x <= usable_min_x or usable_max_y <= usable_min_y:
        logger.warning(
            f"Usable area for seeds is zero or negative ({usable_min_x}-{usable_max_x}, {usable_min_y}-{usable_max_y}). No seeds will be generated.")
        return []

    points: List[Tuple[float, float]] = []
    max_attempts = max_tubuli * 100  # Allow many attempts
    attempts = 0

    while len(points) < max_tubuli and attempts < max_attempts:
        candidate_x = np.random.uniform(usable_min_x, usable_max_x)
        candidate_y = np.random.uniform(usable_min_y, usable_max_y)
        candidate_point = (candidate_x, candidate_y)
        is_valid = True
        for existing_point in points:
            dist = distance.euclidean(candidate_point, existing_point)
            if dist < min_dist:
                is_valid = False
                logger.debug(
                    f"  Rejected candidate point {candidate_point} due to min_dist violation (dist: {dist:.2f} < {min_dist}).")
                break
        if is_valid:
            points.append(candidate_point)
            logger.debug(f"  Accepted candidate point {candidate_point}. Total points: {len(points)}.")
        attempts += 1

    if attempts >= max_attempts and len(points) < max_tubuli:
        logger.warning(
            f"Reached max attempts ({max_attempts}) before finding all tubuli. Generated {len(points)} out of {max_tubuli} requested."
        )
    logger.info(f"Found {len(points)} random seeds within specified constraints.")

    seeds: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for x, y in points:
        slope = np.random.uniform(-1.5, 1.5)
        intercept = y - slope * x
        seeds.append(((slope, intercept), (x, y)))
    logger.debug(f"Formatted {len(seeds)} seeds with slope/intercept information.")
    return seeds


def compute_vignette(cfg: SyntheticDataConfig) -> np.ndarray:
    """
    Computes a 2D array representing a radial vignetting effect.

    Args:
        cfg (SyntheticDataConfig): Configuration object containing vignetting_strength and img_size.

    Returns:
        np.ndarray: A 2D float array (H, W) with values from 0.5 to 1.0, where 1.0 is brightest (center).
    """
    if cfg.vignetting_strength <= 0.0:
        logger.debug("Skipping vignetting (strength is 0 or negative).")
        return 1.0  # Return a scalar 1.0 which will broadcast

    logger.debug(f"Computing vignette with strength={cfg.vignetting_strength:.2f} for image size {cfg.img_size}.")
    try:
        yy, xx = np.mgrid[:cfg.img_size[0], :cfg.img_size[1]]
        # Normalize coordinates to range from -1 to 1 from center
        norm_x = (xx - cfg.img_size[1] / 2) / (cfg.img_size[1] / 2)
        norm_y = (yy - cfg.img_size[0] / 2) / (cfg.img_size[0] / 2)

        # Simple quadratic falloff
        vignette = 1.0 - cfg.vignetting_strength * (norm_x ** 2 + norm_y ** 2)

        vignette = np.clip(vignette, 0.5, 1.0)  # Clamp values to a reasonable range
        logger.debug(f"Vignette computed. Min value: {vignette.min():.4f}, Max value: {vignette.max():.4f}.")
        return vignette
    except Exception as e:
        logger.error(f"Error computing vignette: {e}", exc_info=True)
        return 1.0  # Return neutral value on error


def draw_gaussian_line_rgb(
        frame: np.ndarray,
        mask: Optional[np.ndarray],  # Made optional explicitly
        start_pt: np.ndarray,
        end_pt: np.ndarray,
        sigma_x: float,
        sigma_y: float,
        color_contrast_rgb: Tuple[float, float, float],
        mask_idx: int,
        additional_mask: Optional[np.ndarray] = None,  # Made optional explicitly
):
    """
    Rasterizes a line by placing small 2D Gaussian spots at regular intervals.
    Modifies `frame` and `mask` in-place.
    """
    logger.debug(
        f"Drawing Gaussian line from {start_pt.tolist()} to {end_pt.tolist()}. Sigmas: ({sigma_x:.2f}, {sigma_y:.2f}). Contrast: {color_contrast_rgb}.")

    H, W, C = frame.shape
    if C != 3:
        msg = f"Frame must be a 3-channel RGB image, but got {C} channels (shape: {frame.shape})."
        logger.error(msg)
        raise ValueError(msg)

    # Pre-compute coordinate grids and line properties
    yy, xx = np.mgrid[0:H, 0:W]
    x0, y0 = float(start_pt[0]), float(start_pt[1])
    x1, y1 = float(end_pt[0]), float(end_pt[1])
    vec = np.array([x1 - x0, y1 - y0], dtype=np.float32)
    length = np.linalg.norm(vec)
    mask_threshold = 0.01  # Pixel intensity threshold for mask update

    # Validate sigmas
    if sigma_x <= 0 or sigma_y <= 0:
        logger.warning(
            f"Sigma_x ({sigma_x}) or sigma_y ({sigma_y}) is non-positive. Setting to a small value (1e-6) to prevent division by zero in Gaussian calculation.")
        sigma_x = max(sigma_x, 1e-6)
        sigma_y = max(sigma_y, 1e-6)

    # --- Nested Helper Function for Drawing a Single Spot ---
    def draw_spot_at_point(px: float, py: float):
        """Calculates and applies a single Gaussian spot centered at (px, py)."""
        try:
            # Calculate distances of all grid points from the spot's center
            dx = xx - px
            dy = yy - py

            # Calculate the 2D Gaussian blob
            gaussian_blob = np.exp(-((dx ** 2) / (2 * sigma_x ** 2) + (dy ** 2) / (2 * sigma_y ** 2)))

            # Apply the contrast to each RGB channel and add it to the frame
            for c in range(3):
                frame[..., c] += color_contrast_rgb[c] * gaussian_blob

            # Update the instance mask where the Gaussian is strong enough
            if mask is not None:
                mask[gaussian_blob > mask_threshold] = mask_idx
            if additional_mask is not None:
                additional_mask[gaussian_blob > mask_threshold] = mask_idx
            # logger.debug(f"  Spot drawn at ({px:.2f}, {py:.2f}). Max Gaussian: {gaussian_blob.max():.4f}.")
        except Exception as e:
            logger.error(f"Error drawing spot at ({px:.2f}, {py:.2f}): {e}", exc_info=True)

    # --- End of Helper Function ---

    # If the line has no length, just draw a single spot at the start point.
    if length < 1e-6:
        logger.debug("Line length is negligible. Drawing a single spot.")
        draw_spot_at_point(x0, y0)
        return

    # For lines with length, iterate along the line and draw spots at each step.
    step = 0.5  # Draw a spot every half a pixel for smooth coverage
    num_steps = int(np.ceil(length / step))
    if num_steps == 0:  # Handle cases where length is very small but not zero.
        num_steps = 1

    for i in range(num_steps + 1):
        t = i / num_steps
        px = x0 + t * vec[0]
        py = y0 + t * vec[1]
        draw_spot_at_point(px, py)
    logger.debug(f"Line drawn using {num_steps + 1} spots.")


def annotate_frame(frame: np.ndarray, cfg: SyntheticDataConfig, frame_idx: int) -> np.ndarray:
    """Annotates the frame using the color from the config."""
    logger.debug(f"Annotating frame {frame_idx} (show_time={cfg.show_time}, show_scale={cfg.show_scale}).")
    annotated = frame.copy()
    # Convert 0-1 float color to 0-255 uint8 BGR tuple for OpenCV.
    color_bgr = tuple(int(c * 255) for c in reversed(cfg.annotation_color_rgb))

    H, W = annotated.shape[:2]

    # Time annotation
    if cfg.show_time:
        time_sec = frame_idx / cfg.fps
        time_str = f"{int(time_sec):d}:{int((time_sec % 1) * 100):02d}"  # Format seconds as 00-99 for hundredths
        try:
            cv2.putText(
                annotated, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 2, cv2.LINE_AA
            )
            logger.debug(f"Time annotation '{time_str}' added.")
        except Exception as e:
            logger.error(f"Error adding time annotation for frame {frame_idx}: {e}", exc_info=True)

    # Scale bar
    if cfg.show_scale:
        scale_length_px = int(cfg.scale_bar_um / cfg.um_per_pixel)
        bar_height = 6
        x_end, y_start = W - 10, H - 20
        x_start, y_end = x_end - scale_length_px, y_start - bar_height

        if x_start < 0 or y_end < 0:  # Prevent drawing outside image if scale bar too big
            logger.warning(f"Scale bar would be out of bounds ({x_start},{y_end}) for image size ({W},{H}). Skipping.")
        else:
            try:
                cv2.rectangle(annotated, (x_start, y_end), (x_end, y_start), color_bgr, -1)
                logger.debug(f"Scale bar ({cfg.scale_bar_um}um / {scale_length_px}px) added.")
            except Exception as e:
                logger.error(f"Error adding scale bar for frame {frame_idx}: {e}", exc_info=True)

    return annotated


def build_albumentations_pipeline(cfg: Optional[Union[AlbumentationsConfig, dict]]) -> Optional[A.Compose]:
    """
    Constructs an Albumentations composition from the configuration.
    This pipeline should be applied to the final uint8 frame and mask.
    """
    logger.info("Building Albumentations pipeline.")

    if cfg is None:
        logger.info("Albumentations config is None. Returning None pipeline.")
        return None

    # Helper function to get attribute/item from config (dict or object)
    get_param = lambda key, default: cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)

    transforms: List[A.BasicTransform] = []

    try:
        # --- Geometric transforms ---
        if get_param("horizontal_flip_p", 0.0) > 0:
            transforms.append(A.HorizontalFlip(p=get_param("horizontal_flip_p", 0.0)))
            logger.debug(f"Added HorizontalFlip (p={get_param('horizontal_flip_p', 0.0):.2f}).")

        if get_param("vertical_flip_p", 0.0) > 0:
            transforms.append(A.VerticalFlip(p=get_param("vertical_flip_p", 0.0)))
            logger.debug(f"Added VerticalFlip (p={get_param('vertical_flip_p', 0.0):.2f}).")

        # Note: Albumentations v1.x uses `rotate` for range. v0.x uses `limit`.
        # Assuming v1.x or higher where `rotate` is `limit`.
        # Also, check `shift_scale_rotate_p` which typically combines these.
        # This code uses `rotate_limit` and `affine_p`.
        if get_param("rotate_limit", 0) > 0 or get_param("affine_p", 0.0) > 0.0:
            # Affine is more general and often replaces individual rotation/scale/shear.
            # If `rotate_limit` is the only active parameter here, ensure it's mapped.
            transforms.append(A.Affine(
                rotate=get_param("rotate_limit", 0),
                scale={
                    "scale_limit": get_param("scale_limit", (1.0, 1.0)) if hasattr(cfg, "scale_limit") else (1.0, 1.0)},
                # If no scale, ensure 1.0, 1.0 range
                shear={"x": get_param("shear_limit_x", (0.0, 0.0)) if hasattr(cfg, "shear_limit_x") else (0.0, 0.0),
                       "y": get_param("shear_limit_y", (0.0, 0.0)) if hasattr(cfg, "shear_limit_y") else (0.0, 0.0)},
                p=get_param("shift_scale_rotate_p", 0.5),  # Use common p for this
                fill=0,  # Fill value for pixels outside the boundaries
                mask_fill=0,  # Also fill for masks
                border_mode=cv2.BORDER_CONSTANT
            ))
            logger.debug(
                f"Added Affine (rotate={get_param('rotate_limit', 0)}, p={get_param('shift_scale_rotate_p', 0.5):.2f}).")

        if get_param("elastic_p", 0.0) > 0:
            transforms.append(A.ElasticTransform(
                p=get_param("elastic_p", 0.0),
                alpha=get_param("elastic_alpha", 1),
                sigma=get_param("elastic_sigma", 20),
                alpha_affine=get_param("elastic_alpha_affine", 20),  # Add alpha_affine if present in config
                border_mode=cv2.BORDER_CONSTANT,
                fill_value=0,  # For pixel-level fills
                mask_fill_value=0  # For mask fills
            ))
            logger.debug(
                f"Added ElasticTransform (p={get_param('elastic_p', 0.0):.2f}, alpha={get_param('elastic_alpha', 1)}, sigma={get_param('elastic_sigma', 20)}).")

        if get_param("grid_distortion_p", 0.0) > 0:
            transforms.append(A.GridDistortion(
                p=get_param("grid_distortion_p", 0.0),
                border_mode=cv2.BORDER_CONSTANT,
                fill_value=0,
                mask_fill_value=0
            ))
            logger.debug(f"Added GridDistortion (p={get_param('grid_distortion_p', 0.0):.2f}).")

        # --- Pixel-level & Noise Transforms ---
        if get_param("brightness_contrast_p", 0.0) > 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=get_param("brightness_limit", 0.1),
                contrast_limit=get_param("contrast_limit", 0.1),
                p=get_param("brightness_contrast_p", 0.0)
            ))
            logger.debug(f"Added RandomBrightnessContrast (p={get_param('brightness_contrast_p', 0.0):.2f}).")

        if get_param("gauss_noise_p", 0.0) > 0:
            # Albumentations expects std_limit, not std_range for GaussNoise (a tuple)
            gauss_std_range = get_param("gauss_noise_std_range", (0.05, 0.1))
            gauss_mean_range = get_param("gauss_noise_mean_range", (0.0, 0.0))
            if not isinstance(gauss_std_range, tuple) or len(gauss_std_range) != 2:
                logger.warning(
                    f"gauss_noise_std_range '{gauss_std_range}' is not a 2-element tuple. Skipping GaussNoise.")
            else:
                transforms.append(A.GaussNoise(
                    var_limit=(gauss_std_range[0] ** 2, gauss_std_range[1] ** 2),  # Albumentations uses var_limit
                    mean=get_param("gauss_noise_mean_range", (0.0, 0.0)),  # Here it's a range for mean
                    per_channel=True,
                    p=get_param("gauss_noise_p", 0.0)
                ))
                logger.debug(
                    f"Added GaussNoise (p={get_param('gauss_noise_p', 0.0):.2f}, std_range={gauss_std_range}).")

        master_p = get_param("p", 0.75)  # Master probability for the entire compose pipeline

        if not transforms:
            logger.info("No individual Albumentations transforms configured. Returning None pipeline.")
            return None

        if master_p <= 0.0:
            logger.info(
                f"Master Albumentations probability (p={master_p:.2f}) is zero or negative. Returning None pipeline.")
            return None

        pipeline = A.Compose(transforms, p=master_p)
        logger.info(
            f"Albumentations pipeline built with {len(transforms)} transforms and master probability {master_p:.2f}.")
        return pipeline

    except Exception as e:
        logger.error(f"Error building Albumentations pipeline: {e}. Returning None pipeline.", exc_info=True)
        return None