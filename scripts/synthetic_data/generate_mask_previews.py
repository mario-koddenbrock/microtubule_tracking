import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

def colorize_instance_mask(mask: np.ndarray, cmap_name: str = "tab20") -> np.ndarray:
    """
    Convert an instance segmentation mask into an RGB image
    with distinct colors for each instance ID.

    Args:
        mask: (H,W) array of ints, where 0 = background, 1..N = instances
        cmap_name: matplotlib colormap name (e.g. 'tab20', 'nipy_spectral')

    Returns:
        rgb: (H,W,3) uint8 image
    """
    max_id = mask.max()
    cmap = plt.get_cmap(cmap_name, max_id + 1)

    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for inst_id in range(1, max_id + 1):  # skip background
        rgb[mask == inst_id] = np.array(cmap(inst_id)[:3]) * 255

    return rgb

def process_mask_folder(input_folder: str, output_folder: str, cmap_name: str = "tab20"):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if fname.endswith(".png"):
            mask = np.array(Image.open(os.path.join(input_folder, fname)))
            if mask.ndim == 3:  # handle RGB masks by using only one channel
                mask = mask[:, :, 0]

            rgb = colorize_instance_mask(mask, cmap_name=cmap_name)
            out_path = os.path.join(output_folder, fname)
            Image.fromarray(rgb).save(out_path)
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    input_folder = "data/SynMT/synthetic/validation/image_masks"
    output_folder = "data/SynMT/synthetic/validation/image_masks_preview"
    cmap_name = "plasma"   # try 'tab20', 'viridis', 'plasma', etc.
    process_mask_folder(input_folder, output_folder, cmap_name)
