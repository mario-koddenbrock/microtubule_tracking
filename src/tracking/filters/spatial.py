import logging

import numpy as np
from skimage.measure import label, regionprops

from .base import BaseFilter

logger = logging.getLogger(f"microtuble_tracking.{__name__}")


class CornerExclusionFilter(BaseFilter):
    """
    A filter to exclude regions in the corners of a mask by setting them to zero.

    This is particularly useful for removing static annotations, timestamps, or
    watermarks that may be present in the corners of a video.
    """

    def __init__(self, height_fraction: float = 0.1, width_fraction: float = 0.25):
        """
        Initializes the corner exclusion filter.

        Args:
            height_fraction (float): The fraction of the total image height to
                                     define the corner exclusion zone.
            width_fraction (float): The fraction of the total image width to
                                    define the corner exclusion zone.
        """
        super().__init__()  # Call the base class constructor
        logger.info(
            f"Initializing CornerExclusionFilter with height_fraction={height_fraction}, width_fraction={width_fraction}.")

        if not (0 <= height_fraction < 0.5 and 0 <= width_fraction < 0.5):
            msg = f"Fractions must be between 0.0 and 0.5. Got height_fraction={height_fraction}, width_fraction={width_fraction}."
            logger.error(msg)
            raise ValueError(msg)

        self.height_fraction = height_fraction
        self.width_fraction = width_fraction
        logger.debug("CornerExclusionFilter parameters validated.")

    def filter(self, mask: np.ndarray) -> np.ndarray:
        """
        Filters the mask by completely removing any object instance whose
        bounding box overlaps with the defined corner regions.

        This is more robust than simple slicing as it removes the entire
        object, preventing objects from being "cut" by the filter.

        Args:
            mask (np.ndarray): The input 2D integer label mask.

        Returns:
            np.ndarray: A new mask containing only the objects that do not
                        touch the corner exclusion zones.
        """
        # Leverage base class validation and initial logging
        super().filter(mask)

        num_objects_before = len(np.unique(mask)) - 1  # Count objects before filtering (excluding background 0)
        logger.info(f"Applying CornerExclusionFilter to mask with {num_objects_before} objects.")

        if np.max(mask) == 0:
            logger.info("Input mask is empty (all zeros). Returning as is.")
            return mask  # Return immediately if the mask is empty

        h, w = mask.shape

        # Define the corner boundaries
        corner_h = int(h * self.height_fraction)
        corner_w = int(w * self.width_fraction)
        logger.debug(f"Image dimensions: H={h}, W={w}. Corner exclusion zone: H={corner_h}px, W={corner_w}px.")

        # Create a new mask to populate with only the "good" objects
        filtered_mask = np.zeros_like(mask, dtype=mask.dtype)  # Preserve original mask dtype

        # Use regionprops to get properties of each unique object (instance)
        # `label` ensures the mask has contiguous integers starting from 1.
        labeled_mask = label(mask)
        properties = regionprops(labeled_mask)

        objects_kept = 0
        objects_filtered_out = 0

        for prop_idx, prop in enumerate(properties):
            obj_label = prop.label
            min_r, min_c, max_r, max_c = prop.bbox
            logger.debug(
                f"Processing object ID {obj_label} (original index {prop_idx}). Bounding box: ({min_r},{min_c},{max_r},{max_c}).")

            # Condition for being in the top-left corner
            in_top_left = (min_c < corner_w) and (min_r < corner_h)

            # Condition for being in the bottom-right corner
            in_bottom_right = (max_c > w - corner_w) and (max_r > h - corner_h)

            logger.debug(
                f"  Object {obj_label}: In top-left corner: {in_top_left}, In bottom-right corner: {in_bottom_right}.")

            # --- If the object is NOT in EITHER corner, we keep it ---
            if not in_top_left and not in_bottom_right:
                filtered_mask[labeled_mask == obj_label] = obj_label
                objects_kept += 1
                logger.debug(f"  Object {obj_label} passed corner exclusion. Kept.")
            else:
                objects_filtered_out += 1
                logger.debug(f"  Object {obj_label} failed corner exclusion. Filtered out.")

        num_objects_after = len(np.unique(filtered_mask)) - 1
        logger.info(
            f"CornerExclusionFilter complete. Objects before: {num_objects_before}, after: {num_objects_after}.")
        logger.debug(f"Total objects kept: {objects_kept}, total objects filtered out: {objects_filtered_out}.")

        # Removed the commented-out plt.imshow blocks
        return filtered_mask