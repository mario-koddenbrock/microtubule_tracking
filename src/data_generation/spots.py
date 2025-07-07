import logging
from typing import Tuple, List, Union

import cv2
import numpy as np

from config.spots import SpotConfig

logger = logging.getLogger(f"mt.{__name__}")


class SpotGenerator:
    """
    Manages the state and drawing of a collection of spots (fixed or moving).
    """

    def __init__(self, spot_cfg: SpotConfig, img_shape: Tuple[int, int], color_mode: str = "dark"):
        """
        Initializes the generator and all its spot properties.

        Args:
            spot_cfg (SpotConfig): Configuration for the spots.
            img_shape (Tuple[int, int]): The (height, width) of the image.
            color_mode (str): The color mode ('dark' or 'bright') for spots relative to background.
        """
        logger.info(
            f"Initializing SpotGenerator with {spot_cfg.count} spots, image shape {img_shape}, color mode '{color_mode}'.")
        self.cfg = spot_cfg
        self.img_shape = img_shape
        self.n_spots = self.cfg.count
        self.color_mode = color_mode  # This dictates how spots modify image intensity

        if self.cfg.color_mode != self.color_mode:
            logger.warning(
                f"SpotConfig color_mode '{self.cfg.color_mode}' differs from generator's color_mode '{self.color_mode}'. Using generator's mode for drawing.")

        self._initialize_properties()
        logger.debug(f"SpotGenerator initialized. Total spots: {self.n_spots}.")

    def _initialize_properties(self):
        """Generates the initial state for all spot properties."""
        logger.debug(f"Initializing properties for {self.n_spots} spots.")
        if self.n_spots == 0:
            self.coords, self.intensities, self.radii, self.kernel_sizes = [], [], [], []
            logger.info("No spots to initialize (n_spots is 0).")
            return

        h, w = self.img_shape
        self.coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(self.n_spots)]
        self.intensities = [np.random.uniform(self.cfg.intensity_min, self.cfg.intensity_max) for _ in
                            range(self.n_spots)]
        self.radii = [np.random.randint(self.cfg.radius_min, self.cfg.radius_max + 1) for _ in range(self.n_spots)]
        self.kernel_sizes = [np.random.randint(self.cfg.kernel_size_min, self.cfg.kernel_size_max + 1) for _ in
                             range(self.n_spots)]

        if self.n_spots > 0:
            logger.debug(
                f"Sample initial spot properties: Coords={self.coords[0]}, Intensity={self.intensities[0]:.4f}, Radius={self.radii[0]}, Kernel Size={self.kernel_sizes[0]}.")
        logger.debug("Spot properties initialized.")

    def update(self):
        """
        Updates the state for the next frame. Only moves spots if max_step is defined.
        """
        logger.debug(f"Updating spot positions. Max step config: {self.cfg.max_step}.")
        if self.cfg.max_step is None or self.n_spots == 0:
            logger.debug("Spots are fixed or there are no spots. Skipping position update.")
            return

        h, w = self.img_shape
        step_x = np.random.randint(-self.cfg.max_step, self.cfg.max_step + 1, size=self.n_spots)
        step_y = np.random.randint(-self.cfg.max_step, self.cfg.max_step + 1, size=self.n_spots)

        new_coords = []
        for i, (y, x) in enumerate(self.coords):
            new_y = np.clip(y + step_y[i], 0, h - 1)
            new_x = np.clip(x + step_x[i], 0, w - 1)
            new_coords.append((new_y, new_x))
        self.coords = new_coords
        logger.debug(f"Spots positions updated for {self.n_spots} spots. Sample new coord: {self.coords[0]}.")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Draws the current spots onto the given image."""
        logger.debug(f"Applying {self.n_spots} persistent spots to image of shape {img.shape}.")
        if self.n_spots == 0:
            logger.debug("No persistent spots to draw. Returning original image.")
            return img

        return self.draw_spots(
            img,
            self.coords,
            self.intensities,
            self.radii,
            self.kernel_sizes,
            self.cfg.sigma,
            color_mode=self.color_mode  # Use the generator's color mode
        )

    @staticmethod
    def draw_spots(img: np.ndarray, spot_coords: List[Tuple[int, int]], intensity: Union[float, List[float]],
                   radii: List[int], kernel_sizes: List[int], sigma: float, color_mode: str = "dark") -> np.ndarray:
        """
        Draws spots onto an image. Can handle grayscale or RGB images.

        Args:
            img (np.ndarray): The image to draw on (modified in place or a copy).
            spot_coords (List[Tuple[int, int]]): List of (y, x) coordinates for each spot.
            intensity (Union[float, List[float]]): Intensity value(s) for the spots.
            radii (List[int]): List of radii for each spot.
            kernel_sizes (List[int]): List of kernel sizes for Gaussian blur for each spot.
            sigma (float): Standard deviation for Gaussian blur for all spots.
            color_mode (str): 'dark' to subtract intensity, 'bright' to add intensity.

        Returns:
            np.ndarray: The image with spots drawn.
        """
        logger.debug(f"Drawing {len(spot_coords)} spots on image (shape: {img.shape}, color mode: '{color_mode}').")

        if len(spot_coords) == 0:
            logger.debug("No spot coordinates provided for drawing. Returning original image.")
            return img

        if img.ndim == 3:
            h, w, _ = img.shape
            is_rgb = True
        else:
            h, w = img.shape
            is_rgb = False
        logger.debug(f"Image dimensions: H={h}, W={w}. Is RGB: {is_rgb}.")

        if isinstance(intensity, list):
            if len(intensity) != len(spot_coords):
                msg = f"Intensity list length ({len(intensity)}) must match the length of spot coordinates ({len(spot_coords)})."
                logger.error(msg)
                raise ValueError(msg)
        else:
            intensity = [intensity] * len(spot_coords)

        # Create a copy to avoid modifying the original image directly if needed elsewhere,
        # or just work in place if the caller expects it. Assuming in-place modification is okay.
        # img_out = img.copy() # Uncomment if you need a copy

        for idx, (y, x) in enumerate(spot_coords):
            try:
                # Create a 2D (grayscale) mask for the spot. This is efficient.
                mask = np.zeros((h, w), dtype=np.float32)
                # Draw filled circle
                cv2.circle(mask, (int(x), int(y)), radii[idx], intensity[idx], -1)

                kernel = 2 * kernel_sizes[idx] + 1
                if kernel <= 0:  # Ensure kernel is positive and odd
                    logger.warning(f"Kernel size for spot {idx} is non-positive ({kernel_sizes[idx]}). Forcing to 1.")
                    kernel = 1
                elif kernel % 2 == 0:  # Ensure kernel is odd
                    kernel += 1
                    logger.debug(f"Kernel size for spot {idx} ({kernel_sizes[idx]}) was even, adjusted to {kernel}.")

                # Blur the 2D mask
                if sigma > 0 and kernel > 1:
                    mask = cv2.GaussianBlur(mask, (kernel, kernel), sigma)
                    logger.debug(
                        f"Spot {idx}: Applied Gaussian blur with kernel {kernel}x{kernel} and sigma {sigma:.2f}.")
                else:
                    logger.debug(f"Spot {idx}: Skipping Gaussian blur (sigma={sigma:.2f}, kernel={kernel}).")

                # Add/subtract the mask to the image.
                if is_rgb:
                    # Use NumPy broadcasting to apply the 2D mask to each of the 3 color channels.
                    # `mask[..., np.newaxis]` changes its shape from (H, W) to (H, W, 1),
                    # which NumPy then automatically applies to all 3 channels of `img`.
                    if color_mode == "dark":
                        img -= mask[..., np.newaxis]
                        logger.debug(f"Spot {idx}: Subtracted dark spot.")
                    else:  # "bright"
                        img += mask[..., np.newaxis]
                        logger.debug(f"Spot {idx}: Added bright spot.")
                else:
                    # image is grayscale
                    if color_mode == "dark":
                        img -= mask
                        logger.debug(f"Spot {idx}: Subtracted dark spot (grayscale).")
                    else:  # "bright"
                        img += mask
                        logger.debug(f"Spot {idx}: Added bright spot (grayscale).")
            except Exception as e:
                logger.error(f"Error drawing spot {idx} at ({y},{x}): {e}", exc_info=True)
                # Continue to next spot if one fails, or re-raise if fatal.

        logger.debug(f"Finished drawing {len(spot_coords)} spots.")
        return img

    @staticmethod
    def apply_random_spots(img: np.ndarray, spot_cfg: SpotConfig) -> np.ndarray:
        """
        Adds stateless spots that are regenerated completely on every frame.
        This function is now RGB-aware.

        Args:
            img (np.ndarray): The image to draw on.
            spot_cfg (SpotConfig): Configuration for the random spots.

        Returns:
            np.ndarray: The image with random spots drawn.
        """
        logger.info(f"Applying {spot_cfg.count} random spots to image of shape {img.shape}.")
        n_spots = spot_cfg.count
        if n_spots == 0:
            logger.info("No random spots to apply. Returning original image.")
            return img

        if img.ndim == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape
        logger.debug(f"Image dimensions for random spots: H={h}, W={w}.")

        # Generate all properties on-the-fly for each frame
        coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(n_spots)]
        intensities = [np.random.uniform(spot_cfg.intensity_min, spot_cfg.intensity_max) for _ in range(n_spots)]
        radii = [np.random.randint(spot_cfg.radius_min, spot_cfg.radius_max + 1) for _ in range(n_spots)]
        kernel_sizes = [np.random.randint(spot_cfg.kernel_size_min, spot_cfg.kernel_size_max + 1) for _ in
                        range(n_spots)]

        if n_spots > 0:
            logger.debug(
                f"Sample random spot properties: Coords={coords[0]}, Intensity={intensities[0]:.4f}, Radius={radii[0]}, Kernel Size={kernel_sizes[0]}.")

        # This call is now safe because SpotGenerator.draw_spots is RGB-aware.
        return SpotGenerator.draw_spots(
            img,
            coords,
            intensities,
            radii,
            kernel_sizes,
            spot_cfg.sigma,
            color_mode=spot_cfg.color_mode,  # Use the config's color mode for random spots
        )