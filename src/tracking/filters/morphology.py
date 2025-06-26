import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from typing import Optional

from .base import BaseFilter


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
        self.min_area = min_area
        self.max_area = max_area if max_area is not None else float('inf')
        self.min_aspect_ratio = min_aspect_ratio
        self.min_solidity = min_solidity
        print(
            "MorphologyFilter initialized with: "
            f"min_area={min_area}, max_area={self.max_area}, "
            f"min_aspect_ratio={min_aspect_ratio}, min_solidity={min_solidity}"
        )

    def filter(self, mask: np.ndarray) -> np.ndarray:
        """
        Filters the given mask, keeping only objects that match the criteria.
        """
        if np.max(mask) == 0:
            return mask  # Return empty mask if it's already empty

        # Ensure mask is properly labeled
        labeled_mask = label(mask)

        # Calculate properties for each labeled region
        properties = regionprops(labeled_mask)

        filtered_mask = np.zeros_like(mask)
        mask_rest = np.zeros_like(mask)

        for prop in properties:

            # plt.imshow(mask_rest)
            # plt.title("Rest")
            # plt.axis('off')
            # plt.show()

            # --- Area Check ---
            if not (self.min_area <= prop.area <= self.max_area):
                mask_rest[labeled_mask == prop.label] = 1
                continue

            # --- Solidity Check ---
            if prop.solidity < self.min_solidity:
                mask_rest[labeled_mask == prop.label] = 50
                continue

            # --- Aspect Ratio Check ---
            # Use major and minor axis lengths of the equivalent ellipse
            minor_axis = prop.minor_axis_length
            major_axis = prop.major_axis_length

            # Avoid division by zero for perfectly thin lines
            if minor_axis == 0:
                aspect_ratio = float('inf')
            else:
                aspect_ratio = major_axis / minor_axis

            if aspect_ratio < self.min_aspect_ratio:
                mask_rest[labeled_mask == prop.label] = 100
                continue

            # If all checks pass, keep this object
            filtered_mask[labeled_mask == prop.label] = 150

        # plt.imshow(filtered_mask)
        # plt.title("Filtered Mask")
        # plt.axis('off')
        # plt.show()
        # print("Filtered mask with morphological properties applied.")
        #
        # plt.imshow(mask_rest)
        # plt.title("Rest")
        # plt.axis('off')
        # plt.show()

        return filtered_mask