import matplotlib.pyplot as plt
import numpy as np

from skimage.measure import label, regionprops

from .base import BaseFilter


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
        if not (0 <= height_fraction < 0.5 and 0 <= width_fraction < 0.5):
            raise ValueError("Fractions must be between 0.0 and 0.5.")

        self.height_fraction = height_fraction
        self.width_fraction = width_fraction

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
        if np.max(mask) == 0:
            return mask  # Return immediately if the mask is empty

        h, w = mask.shape

        # Define the corner boundaries
        corner_h = int(h * self.height_fraction)
        corner_w = int(w * self.width_fraction)

        # Create a new mask to populate with only the "good" objects
        filtered_mask = np.zeros_like(mask)

        # Use regionprops to get properties of each unique object (instance)
        # `label` ensures the mask has contiguous integers starting from 1.
        labeled_mask = label(mask)
        properties = regionprops(labeled_mask)

        for prop in properties:
            # Get the bounding box of the current object
            min_r, min_c, max_r, max_c = prop.bbox

            # Condition for being in the top-left corner
            in_top_left = (min_c < corner_w) and (min_r < corner_h)

            # Condition for being in the bottom-right corner
            in_bottom_right = (max_c > w - corner_w) and (max_r > h - corner_h)

            # --- If the object is NOT in EITHER corner, we keep it ---
            if not in_top_left and not in_bottom_right:
                # Add the entire object to our new, clean mask.
                # We use the original label value from the object's properties.
                filtered_mask[labeled_mask == prop.label] = prop.label

                # plt.imshow(filtered_mask)
                # plt.title("Filtered Mask with Corner Exclusion")
                # plt.axis('off')
                # plt.show()
                # print("Filtered mask with corner exclusion applied.")

        # plt.imshow(labeled_mask)
        # plt.title("Filtered Mask with Corner Exclusion")
        # plt.axis('off')
        # plt.show()
        #
        # plt.imshow(filtered_mask)
        # plt.title("Filtered Mask with Corner Exclusion")
        # plt.axis('off')
        # plt.show()


        return filtered_mask