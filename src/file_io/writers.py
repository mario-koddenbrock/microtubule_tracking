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

    def __init__(
        self,
        cfg: SyntheticDataConfig,
        base_output_dir: str,
        export_video: bool = True,
        export_gif_preview: bool = True,
        export_mp4_preview: bool = True,
    ):
        """
        Initializes all file paths and writer objects based on the config.
        """
        self.cfg = cfg
        os.makedirs(base_output_dir, exist_ok=True)

        # --- 1. Define all output paths ---
        base_name = f"series_{cfg.id}"
        video_tiff_path = os.path.join(base_output_dir, f"{base_name}_video.tif")
        tubuli_masks_tiff_path = os.path.join(base_output_dir, f"{base_name}_masks.tif")
        seed_masks_tiff_path = os.path.join(base_output_dir, f"{base_name}_seed_masks.tif")
        video_mp4_path = os.path.join(base_output_dir, f"{base_name}_video_preview.mp4")
        tubuli_masks_mp4_path = os.path.join(base_output_dir, f"{base_name}_masks_preview.mp4")
        seed_masks_mp4_path = os.path.join(base_output_dir, f"{base_name}_seed_masks_preview.mp4")
        gif_path = os.path.join(base_output_dir, f"{base_name}_video_preview.gif")

        self.export_video = export_video
        self.export_gif_preview = export_gif_preview
        self.export_mp4_preview = export_mp4_preview

        # --- 2. Initialize writers ---
        if self.export_video:
            self.video_tiff_writer = imageio.get_writer(video_tiff_path, format="TIFF")
        else:
            self.video_tiff_writer = None

        if self.export_gif_preview:
            self.gif_writer = imageio.get_writer(gif_path, fps=cfg.fps)
        else:
            self.gif_writer = None

        if self.export_mp4_preview:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            img_h, img_w = cfg.img_size
            self.video_mp4_writer = cv2.VideoWriter(video_mp4_path, fourcc, cfg.fps, (img_w, img_h))
        else:
            self.video_mp4_writer = None

        if cfg.generate_tubuli_mask:
            self.tubuli_mask_tiff_writer = imageio.get_writer(tubuli_masks_tiff_path, format="TIFF")
            self.tubuli_mask_mp4_writer = cv2.VideoWriter(
                tubuli_masks_mp4_path, fourcc, cfg.fps, (img_w, img_h)
            )
        else:
            self.tubuli_mask_tiff_writer = None
            self.tubuli_mask_mp4_writer = None

        if cfg.generate_seed_mask:
            # Since this is "fixed", it will only be one frame.
            self.seed_mask_tiff_writer = imageio.get_writer(seed_masks_tiff_path, format="TIFF")
            # Still, we want to have a colorful vis
            self.seed_mask_mp4_writer = cv2.VideoWriter(
                seed_masks_mp4_path, fourcc, cfg.fps, (img_w, img_h)
            )
        else:
            self.seed_mask_tiff_writer = None
            self.seed_mask_mp4_writer = None

    def append(
        self, frame_img_rgb: np.ndarray, tubuli_mask_img: np.ndarray, seed_mask_img: np.ndarray
    ):
        """
        Appends a new frame and its mask to all relevant output files.
        """
        # A. Write the main video frames (which are already uint8 RGB)
        if self.video_tiff_writer:
            self.video_tiff_writer.append_data(frame_img_rgb)

        if self.gif_writer:
            self.gif_writer.append_data(frame_img_rgb)

        # Convert RGB to BGR for OpenCV's VideoWriter
        frame_bgr = cv2.cvtColor(frame_img_rgb, cv2.COLOR_RGB2BGR)
        if self.video_mp4_writer:
            self.video_mp4_writer.write(frame_bgr)

        # B.1. Write the mask frames (if enabled)
        if self.cfg.generate_tubuli_mask and tubuli_mask_img is not None:
            self.tubuli_mask_tiff_writer.append_data(tubuli_mask_img)  # Raw uint16 data

            # Create and write the colorized preview for the mask
            mask_vis_float = label2rgb(tubuli_mask_img, bg_label=0)
            mask_vis_uint8 = (mask_vis_float * 255).astype(np.uint8)
            mask_vis_bgr = cv2.cvtColor(mask_vis_uint8, cv2.COLOR_RGB2BGR)
            self.tubuli_mask_mp4_writer.write(mask_vis_bgr)

        # B.2 Write the seed mask frame (if enabled)
        if self.cfg.generate_seed_mask and seed_mask_img is not None:
            self.seed_mask_tiff_writer.append_data(seed_mask_img)

            # Create and write the colorized preview for the mask
            mask_vis_float = label2rgb(seed_mask_img, bg_label=0)
            mask_vis_uint8 = (mask_vis_float * 255).astype(np.uint8)
            mask_vis_bgr = cv2.cvtColor(mask_vis_uint8, cv2.COLOR_RGB2BGR)
            self.seed_mask_mp4_writer.write(mask_vis_bgr)

    def close(self):
        """Closes all writer objects to finalize files."""
        print("Closing all file writers...")
        if self.video_tiff_writer:
            self.video_tiff_writer.close()
        if self.video_mp4_writer:
            self.video_mp4_writer.release()
        if self.gif_writer:
            self.gif_writer.close()
        if self.tubuli_mask_tiff_writer:
            self.tubuli_mask_tiff_writer.close()
        if self.tubuli_mask_mp4_writer:
            self.tubuli_mask_mp4_writer.release()
        if self.seed_mask_tiff_writer:
            self.seed_mask_tiff_writer.close()
        if self.seed_mask_mp4_writer:
            self.seed_mask_mp4_writer.release()
        print("All files saved successfully.")
