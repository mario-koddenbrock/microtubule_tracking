import logging
from typing import Optional  

import numpy as np
from skimage.measure import label, regionprops

from .base import BaseFilter

logger = logging.getLogger(f"microtuble_tracking.{__name__}")


class MorphologyFilter(BaseFilter):
    """
    Filters mask objects based on morphological properties like area,
    aspect ratio, and solidity. Ideal for selecting elongated, "worm-like" shapes.
    """

    def __init__(
            self,
            min_area: int = 10,
            max_area: Optional[int] = None,
            min_aspect_ratio: float = 3.0,
            min_solidity: float = 0.20,
    ):
        """
        Initializes the morphological filter.

        Args:
            min_area (int): The minimum pixel area an object must have to be kept.
            max_area (Optional[int]): The maximum pixel area. If None, no upper
                                      limit is applied.
            min_aspect_ratio (float): The minimum aspect ratio (length/width)
                                      an object must have.
            min_solidity (float): The minimum solidity (area / convex_hull_area)
                                  an object must have.
        """
        super().__init__()  # Call the base class constructor
        self.min_area = min_area
        self.max_area = max_area if max_area is not None else float('inf')
        self.min_aspect_ratio = min_aspect_ratio
        self.min_solidity = min_solidity

        logger.info(
            f"MorphologyFilter initialized with: "
            f"min_area={min_area}, max_area={self.max_area}, "
            f"min_aspect_ratio={min_aspect_ratio}, min_solidity={min_solidity}"
        )
        # Log specific ranges for debug
        logger.debug(
            f"Area range: [{self.min_area}, {self.max_area}], Aspect Ratio min: {self.min_aspect_ratio}, Solidity min: {self.min_solidity}.")

    def filter(self, mask: np.ndarray) -> np.ndarray:
        """
        Filters the given mask, keeping only objects that match the criteria.
        """
        # Leverage base class validation and initial logging
        super().filter(mask)

        num_objects_before = len(np.unique(mask)) - 1  # Count objects before filtering (excluding background 0)
        logger.info(f"Applying MorphologyFilter to mask with {num_objects_before} objects.")

        if np.max(mask) == 0:
            logger.info("Input mask is empty (all zeros). Returning as is.")
            return mask  # Return empty mask if it's already empty

        # Ensure mask is properly labeled (important if input mask might not have contiguous labels)
        labeled_mask = label(mask)
        # Re-get properties from labeled mask to ensure correct unique IDs
        properties = regionprops(labeled_mask)

        filtered_mask = np.zeros_like(mask, dtype=mask.dtype)  # Preserve original mask dtype

        # Counters for filtered objects
        filtered_area = 0
        filtered_solidity = 0
        filtered_aspect_ratio = 0

        for prop_idx, prop in enumerate(properties):
            obj_label = prop.label
            logger.debug(f"Processing object ID {obj_label} (original index {prop_idx}).")

            # --- Area Check ---
            if not (self.min_area <= prop.area <= self.max_area):
                logger.debug(
                    f"  Object {obj_label} failed area check (area={prop.area}). Expected [{self.min_area}, {self.max_area}]. Filtering out.")
                filtered_area += 1
                continue

            # --- Solidity Check ---
            if prop.solidity < self.min_solidity:
                logger.debug(
                    f"  Object {obj_label} failed solidity check (solidity={prop.solidity:.4f}). Expected >= {self.min_solidity}. Filtering out.")
                filtered_solidity += 1
                continue

            # --- Aspect Ratio Check ---
            # Use major and minor axis lengths of the equivalent ellipse
            minor_axis = prop.minor_axis_length
            major_axis = prop.major_axis_length

            # Avoid division by zero for perfectly thin lines
            if minor_axis == 0:
                aspect_ratio = float('inf')
                logger.debug(f"  Object {obj_label}: Minor axis is zero, aspect ratio set to infinity.")
            else:
                aspect_ratio = major_axis / minor_axis

            if aspect_ratio < self.min_aspect_ratio:
                logger.debug(
                    f"  Object {obj_label} failed aspect ratio check (aspect_ratio={aspect_ratio:.4f}). Expected >= {self.min_aspect_ratio}. Filtering out.")
                filtered_aspect_ratio += 1
                continue

            # If all checks pass, keep this object
            filtered_mask[labeled_mask == obj_label] = obj_label
            logger.debug(f"  Object {obj_label} passed all checks. Kept in mask.")

        num_objects_after = len(np.unique(filtered_mask)) - 1
        logger.info(f"MorphologyFilter complete. Objects before: {num_objects_before}, after: {num_objects_after}.")
        logger.debug(
            f"Filtered out: {filtered_area} by area, {filtered_solidity} by solidity, {filtered_aspect_ratio} by aspect ratio.")

        # Removed the commented-out plt.imshow blocks
        return filtered_mask