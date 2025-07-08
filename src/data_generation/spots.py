import logging
from typing import Tuple, List, Union

import cv2
import numpy as np

from config.spots import SpotConfig

logger = logging.getLogger(f"mt.{__name__}")


def _generate_polygon_vertices(center_y: int, center_x: int, avg_radius: int, min_verts: int, max_verts: int) -> np.ndarray:
    """Helper to generate vertices for a random polygon."""
    num_vertices = np.random.randint(min_verts, max_verts + 1)
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    # Add randomness to angles and radii to create irregular shapes
    angles += np.random.uniform(-0.5, 0.5, num_vertices) * (2 * np.pi / num_vertices)
    radii = np.random.uniform(avg_radius * 0.7, avg_radius * 1.3, num_vertices)

    points = []
    for angle, radius in zip(angles, radii):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append([x, y])

    return np.array(points, dtype=np.int32)


class SpotGenerator:
    """
    Manages the state and drawing of a collection of spots (fixed or moving).
    Now supports circular and polygonal shapes for fixed/moving spots.
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
        self.color_mode = color_mode

        if self.cfg.color_mode != self.color_mode:
            logger.warning(
                f"SpotConfig color_mode '{self.cfg.color_mode}' differs from generator's color_mode '{self.color_mode}'. Using generator's mode for drawing.")

        self._initialize_properties()
        logger.debug(f"SpotGenerator initialized. Total spots: {self.n_spots}.")

    def _initialize_properties(self):
        """Generates the initial state for all spot properties, including shapes."""
        logger.debug(f"Initializing properties for {self.n_spots} spots.")
        if self.n_spots == 0:
            self.coords, self.intensities, self.radii, self.kernel_sizes = [], [], [], []
            self.spot_shapes, self.polygon_vertices = [], {}
            logger.info("No spots to initialize (n_spots is 0).")
            return

        h, w = self.img_shape
        self.coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(self.n_spots)]
        self.intensities = [np.random.uniform(self.cfg.intensity_min, self.cfg.intensity_max) for _ in
                            range(self.n_spots)]
        self.radii = [np.random.randint(self.cfg.radius_min, self.cfg.radius_max + 1) for _ in range(self.n_spots)]
        self.kernel_sizes = [np.random.randint(self.cfg.kernel_size_min, self.cfg.kernel_size_max + 1) for _ in
                             range(self.n_spots)]

        # --- New: Determine shape and pre-calculate polygon vertices ---
        self.spot_shapes = []
        self.polygon_vertices = {}
        for i in range(self.n_spots):
            if np.random.random() < self.cfg.polygon_p:
                self.spot_shapes.append('polygon')
                (y, x) = self.coords[i]
                radius = self.radii[i]
                self.polygon_vertices[i] = _generate_polygon_vertices(
                    y, x, radius,
                    self.cfg.polygon_vertex_count_min,
                    self.cfg.polygon_vertex_count_max
                )
                logger.debug(f"Spot {i} initialized as a polygon with {len(self.polygon_vertices[i])} vertices.")
            else:
                self.spot_shapes.append('circle')
                logger.debug(f"Spot {i} initialized as a circle.")

        if self.n_spots > 0:
            logger.debug(
                f"Sample initial spot properties: Coords={self.coords[0]}, Shape={self.spot_shapes[0]}, Intensity={self.intensities[0]:.4f}, Radius={self.radii[0]}, Kernel Size={self.kernel_sizes[0]}.")
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
            # If it's a polygon, we need to move its vertices too
            if self.spot_shapes[i] == 'polygon':
                self.polygon_vertices[i][:, 0] += step_x[i]
                self.polygon_vertices[i][:, 1] += step_y[i]

        self.coords = new_coords
        if self.n_spots > 0:
            logger.debug(f"Spots positions updated for {self.n_spots} spots. Sample new coord: {self.coords[0]}.")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Draws the current spots onto the given image, supporting both circles and polygons."""
        logger.debug(f"Applying {self.n_spots} persistent spots to image of shape {img.shape}.")
        if self.n_spots == 0:
            logger.debug("No persistent spots to draw. Returning original image.")
            return img

        if img.ndim == 3:
            h, w, _ = img.shape
            is_rgb = True
        else:
            h, w = img.shape
            is_rgb = False

        for idx in range(self.n_spots):
            try:
                # Create a 2D (grayscale) mask for each spot.
                mask = np.zeros((h, w), dtype=np.float32)
                intensity = self.intensities[idx]
                (y, x) = self.coords[idx]

                # Draw the shape onto the mask
                if self.spot_shapes[idx] == 'polygon':
                    verts = self.polygon_vertices[idx]
                    cv2.fillPoly(mask, [verts], intensity)
                else:  # 'circle'
                    radius = self.radii[idx]
                    cv2.circle(mask, (int(x), int(y)), radius, intensity, -1)

                kernel_size = self.kernel_sizes[idx]
                kernel = 2 * kernel_size + 1
                if kernel <= 0:
                    logger.warning(f"Kernel size for spot {idx} is non-positive ({kernel_size}). Forcing to 1.")
                    kernel = 1
                elif kernel % 2 == 0:
                    kernel += 1

                # Blur the 2D mask
                if self.cfg.sigma > 0 and kernel > 1:
                    mask = cv2.GaussianBlur(mask, (kernel, kernel), self.cfg.sigma)

                # Add/subtract the mask to the image.
                if is_rgb:
                    if self.color_mode == "dark":
                        img -= mask[..., np.newaxis]
                    else:  # "bright"
                        img += mask[..., np.newaxis]
                else:
                    if self.color_mode == "dark":
                        img -= mask
                    else:  # "bright"
                        img += mask
            except Exception as e:
                logger.error(f"Error drawing spot {idx} at ({self.coords[idx]}): {e}", exc_info=True)

        logger.debug(f"Finished drawing {self.n_spots} persistent spots.")
        return img

    @staticmethod
    def draw_spots(img: np.ndarray, spot_coords: List[Tuple[int, int]], intensity: Union[float, List[float]],
                   radii: List[int], kernel_sizes: List[int], sigma: float, color_mode: str = "dark") -> np.ndarray:
        """
        Draws circular spots onto an image. Can handle grayscale or RGB images.
        This is kept for stateless random spots which are always circles.
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

        for idx, (y, x) in enumerate(spot_coords):
            try:
                mask = np.zeros((h, w), dtype=np.float32)
                cv2.circle(mask, (int(x), int(y)), radii[idx], intensity[idx], -1)

                kernel = 2 * kernel_sizes[idx] + 1
                if kernel <= 0:
                    logger.warning(f"Kernel size for spot {idx} is non-positive ({kernel_sizes[idx]}). Forcing to 1.")
                    kernel = 1
                elif kernel % 2 == 0:
                    kernel += 1

                if sigma > 0 and kernel > 1:
                    mask = cv2.GaussianBlur(mask, (kernel, kernel), sigma)
                    logger.debug(
                        f"Spot {idx}: Applied Gaussian blur with kernel {kernel}x{kernel} and sigma {sigma:.2f}.")
                else:
                    logger.debug(f"Spot {idx}: Skipping Gaussian blur (sigma={sigma:.2f}, kernel={kernel}).")

                if is_rgb:
                    if color_mode == "dark":
                        img -= mask[..., np.newaxis]
                    else:  # "bright"
                        img += mask[..., np.newaxis]
                else:
                    if color_mode == "dark":
                        img -= mask
                    else:  # "bright"
                        img += mask
            except Exception as e:
                logger.error(f"Error drawing spot {idx} at ({y},{x}): {e}", exc_info=True)

        logger.debug(f"Finished drawing {len(spot_coords)} spots.")
        return img

    @staticmethod
    def apply_random_spots(img: np.ndarray, spot_cfg: SpotConfig) -> np.ndarray:
        """
        Adds stateless circular spots that are regenerated completely on every frame.
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

        coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(n_spots)]
        intensities = [np.random.uniform(spot_cfg.intensity_min, spot_cfg.intensity_max) for _ in range(n_spots)]
        radii = [np.random.randint(spot_cfg.radius_min, spot_cfg.radius_max + 1) for _ in range(n_spots)]
        kernel_sizes = [np.random.randint(spot_cfg.kernel_size_min, spot_cfg.kernel_size_max + 1) for _ in
                        range(n_spots)]

        if n_spots > 0:
            logger.debug(
                f"Sample random spot properties: Coords={coords[0]}, Intensity={intensities[0]:.4f}, Radius={radii[0]}, Kernel Size={kernel_sizes[0]}.")

        return SpotGenerator.draw_spots(
            img,
            coords,
            intensities,
            radii,
            kernel_sizes,
            spot_cfg.sigma,
            color_mode=spot_cfg.color_mode,
        )