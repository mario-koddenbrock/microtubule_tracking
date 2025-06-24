import os
import cv2
import imageio
import numpy as np
from skimage.color import label2rgb

from config.synthetic_data import SyntheticDataConfig


class VideoOutputManager:
    """
    Manages the creation and writing process for all video/image sequence outputs.

    This class centralizes file path generation and writer object handling
    (TIFF, MP4, GIF) to simplify the main video generation loop.
    """

    def __init__(self, cfg: SyntheticDataConfig, base_output_dir: str):
        """
        Initializes all file paths and writer objects based on the config.
        """
        self.cfg = cfg
        os.makedirs(base_output_dir, exist_ok=True)

        # --- 1. Define all output paths ---
        base_name = f"series_{cfg.id}"
        video_tiff_path = os.path.join(base_output_dir, f"{base_name}_video.tif")
        masks_tiff_path = os.path.join(base_output_dir, f"{base_name}_masks.tif")
        video_mp4_path = os.path.join(base_output_dir, f"{base_name}_video_preview.mp4")
        masks_mp4_path = os.path.join(base_output_dir, f"{base_name}_masks_preview.mp4")
        gif_path = os.path.join(base_output_dir, f"{base_name}_video_preview.gif")

        # --- 2. Initialize writers ---
        self.video_tiff_writer = imageio.get_writer(video_tiff_path, format='TIFF')
        self.gif_writer = imageio.get_writer(gif_path, fps=cfg.fps)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        img_h, img_w = cfg.img_size
        self.video_mp4_writer = cv2.VideoWriter(video_mp4_path, fourcc, cfg.fps, (img_w, img_h))

        if cfg.generate_mask:
            self.mask_tiff_writer = imageio.get_writer(masks_tiff_path, format='TIFF')
            self.mask_mp4_writer = cv2.VideoWriter(masks_mp4_path, fourcc, cfg.fps, (img_w, img_h))
        else:
            self.mask_tiff_writer = None
            self.mask_mp4_writer = None

    def append(self, frame_img_rgb: np.ndarray, mask_img: np.ndarray):
        """
        Appends a new frame and its mask to all relevant output files.
        """
        # A. Write the main video frames (which are already uint8 RGB)
        self.video_tiff_writer.append_data(frame_img_rgb)
        self.gif_writer.append_data(frame_img_rgb)

        # Convert RGB to BGR for OpenCV's VideoWriter
        frame_bgr = cv2.cvtColor(frame_img_rgb, cv2.COLOR_RGB2BGR)
        self.video_mp4_writer.write(frame_bgr)

        # B. Write the mask frames (if enabled)
        if self.cfg.generate_mask and mask_img is not None:
            self.mask_tiff_writer.append_data(mask_img)  # Raw uint16 data

            # Create and write the colorized preview for the mask
            mask_vis_float = label2rgb(mask_img, bg_label=0)
            mask_vis_uint8 = (mask_vis_float * 255).astype(np.uint8)
            mask_vis_bgr = cv2.cvtColor(mask_vis_uint8, cv2.COLOR_RGB2BGR)
            self.mask_mp4_writer.write(mask_vis_bgr)

    def close(self):
        """Closes all writer objects to finalize files."""
        print("Closing all file writers...")
        self.video_tiff_writer.close()
        self.gif_writer.close()
        self.video_mp4_writer.release()
        if self.mask_tiff_writer:
            self.mask_tiff_writer.close()
        if self.mask_mp4_writer:
            self.mask_mp4_writer.release()
        print("All files saved successfully.")