import logging
import os
from typing import Optional

import cv2
import imageio
import numpy as np
from skimage.color import label2rgb

from config.synthetic_data import SyntheticDataConfig

logger = logging.getLogger(f"mt.{__name__}")


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
        logger.info(f"Initializing VideoOutputManager for series ID: {cfg.id}, output directory: '{base_output_dir}'")
        self.cfg = cfg

        try:
            os.makedirs(base_output_dir, exist_ok=True)
            logger.debug(f"Ensured base output directory exists: {base_output_dir}")
        except OSError as e:
            logger.critical(f"Failed to create base output directory '{base_output_dir}': {e}", exc_info=True)
            raise  # Re-raise, as we cannot proceed without output directory

        # --- 1. Define all output paths ---
        base_name = f"series_{cfg.id}"
        self.video_tiff_path = os.path.join(base_output_dir, f"{base_name}_video.tif")
        self.microtubule_masks_tiff_path = os.path.join(base_output_dir, f"{base_name}_masks.tif")
        self.seed_masks_tiff_path = os.path.join(base_output_dir, f"{base_name}_seed_masks.tif")
        self.video_mp4_path = os.path.join(base_output_dir, f"{base_name}_video_preview.mp4")
        self.microtubule_masks_mp4_path = os.path.join(base_output_dir, f"{base_name}_masks_preview.mp4")
        self.seed_masks_mp4_path = os.path.join(base_output_dir, f"{base_name}_seed_masks_preview.mp4")
        self.gif_path = os.path.join(base_output_dir, f"{base_name}_video_preview.gif")

        logger.debug(
            f"Output paths defined: {self.video_tiff_path}, {self.microtubule_masks_tiff_path}, {self.seed_masks_tiff_path}, {self.video_mp4_path}, {self.microtubule_masks_mp4_path}, {self.seed_masks_mp4_path}, {self.gif_path}.")

        self.export_video = export_video
        self.export_gif_preview = export_gif_preview
        self.export_mp4_preview = export_mp4_preview

        # --- 2. Initialize writers ---
        img_h, img_w = cfg.img_size
        mp4_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        logger.debug(f"MP4 codec FOURCC: '{mp4_fourcc}'. Expected frame size for MP4: ({img_w}, {img_h}).")

        self.video_tiff_writer: Optional[imageio.core.format.Writer] = None
        if self.export_video:
            try:
                self.video_tiff_writer = imageio.get_writer(self.video_tiff_path, format="TIFF")
                logger.info(f"Initialized TIFF video writer: {self.video_tiff_path}")
            except Exception as e:
                logger.error(f"Failed to initialize TIFF video writer {self.video_tiff_path}: {e}", exc_info=True)

        self.gif_writer: Optional[imageio.core.format.Writer] = None
        if self.export_gif_preview:
            try:
                self.gif_writer = imageio.get_writer(self.gif_path, fps=cfg.fps, mode="I", loop=0)
                logger.info(f"Initialized GIF preview writer: {self.gif_path}")
            except Exception as e:
                logger.error(f"Failed to initialize GIF writer {self.gif_path}: {e}", exc_info=True)

        self.video_mp4_writer: Optional[cv2.VideoWriter] = None
        if self.export_mp4_preview:
            try:
                self.video_mp4_writer = cv2.VideoWriter(self.video_mp4_path, mp4_fourcc, cfg.fps, (img_w, img_h))
                if not self.video_mp4_writer.isOpened():
                    raise IOError("OpenCV VideoWriter failed to open.")
                logger.info(f"Initialized MP4 video writer: {self.video_mp4_path}")
            except Exception as e:
                logger.error(f"Failed to initialize MP4 video writer {self.video_mp4_path}: {e}", exc_info=True)

        self.microtubule_mask_tiff_writer: Optional[imageio.core.format.Writer] = None
        self.microtubule_mask_mp4_writer: Optional[cv2.VideoWriter] = None
        if cfg.generate_microtubule_mask:
            try:
                self.microtubule_mask_tiff_writer = imageio.get_writer(self.microtubule_masks_tiff_path, format="TIFF")
                logger.info(f"Initialized TIFF microtubule mask writer: {self.microtubule_masks_tiff_path}")
                self.microtubule_mask_mp4_writer = cv2.VideoWriter(
                    self.microtubule_masks_mp4_path, mp4_fourcc, cfg.fps, (img_w, img_h)
                )
                if not self.microtubule_mask_mp4_writer.isOpened():
                    raise IOError("OpenCV microtubule Mask VideoWriter failed to open.")
                logger.info(f"Initialized MP4 microtubule mask writer: {self.microtubule_masks_mp4_path}")
            except Exception as e:
                logger.error(
                    f"Failed to initialize microtubule mask writers ({self.microtubule_masks_tiff_path}, {self.microtubule_masks_mp4_path}): {e}",
                    exc_info=True)
        else:
            logger.debug("microtubule mask generation disabled. Skipping writer initialization.")

        self.seed_mask_tiff_writer: Optional[imageio.core.format.Writer] = None
        self.seed_mask_mp4_writer: Optional[cv2.VideoWriter] = None
        if cfg.generate_seed_mask:
            try:
                self.seed_mask_tiff_writer = imageio.get_writer(self.seed_masks_tiff_path, format="TIFF")
                logger.info(f"Initialized TIFF seed mask writer: {self.seed_masks_tiff_path}")
                self.seed_mask_mp4_writer = cv2.VideoWriter(
                    self.seed_masks_mp4_path, mp4_fourcc, cfg.fps, (img_w, img_h)
                )
                if not self.seed_mask_mp4_writer.isOpened():
                    raise IOError("OpenCV Seed Mask VideoWriter failed to open.")
                logger.info(f"Initialized MP4 seed mask writer: {self.seed_masks_mp4_path}")
            except Exception as e:
                logger.error(
                    f"Failed to initialize seed mask writers ({self.seed_masks_tiff_path}, {self.seed_masks_mp4_path}): {e}",
                    exc_info=True)
        else:
            logger.debug("Seed mask generation disabled. Skipping writer initialization.")

        logger.info("VideoOutputManager initialization complete.")

    def append(
            self, frame_img_rgb: np.ndarray, microtubule_mask_img: Optional[np.ndarray], seed_mask_img: Optional[np.ndarray]
    ):
        """
        Appends a new frame and its mask to all relevant output files.
        """
        logger.debug("Appending new frame to output writers...")

        # A. Write the main video frames (which are already uint8 RGB)
        try:
            if self.video_tiff_writer:
                self.video_tiff_writer.append_data(frame_img_rgb)
                logger.debug("Appended frame to TIFF video writer.")
            if self.gif_writer:
                self.gif_writer.append_data(frame_img_rgb)
                logger.debug("Appended frame to GIF writer.")

            if self.video_mp4_writer:
                # Convert RGB to BGR for OpenCV's VideoWriter
                frame_bgr = cv2.cvtColor(frame_img_rgb, cv2.COLOR_RGB2BGR)
                self.video_mp4_writer.write(frame_bgr)
                logger.debug("Appended frame to MP4 video writer.")
        except Exception as e:
            logger.error(f"Error appending main video frame: {e}", exc_info=True)

        # B.1. Write the microtubule mask frames (if enabled)
        if self.cfg.generate_microtubule_mask:
            if microtubule_mask_img is not None:
                try:
                    if self.microtubule_mask_tiff_writer:
                        self.microtubule_mask_tiff_writer.append_data(microtubule_mask_img)  # Raw uint16 data
                        logger.debug("Appended microtubule mask to TIFF writer.")

                    if self.microtubule_mask_mp4_writer:
                        # Create and write the colorized preview for the mask
                        mask_vis_float = label2rgb(microtubule_mask_img, bg_label=0)
                        mask_vis_uint8 = (mask_vis_float * 255).astype(np.uint8)
                        mask_vis_bgr = cv2.cvtColor(mask_vis_uint8, cv2.COLOR_RGB2BGR)
                        self.microtubule_mask_mp4_writer.write(mask_vis_bgr)
                        logger.debug("Appended colorized microtubule mask to MP4 writer.")
                except Exception as e:
                    logger.error(f"Error appending microtubule mask frame: {e}", exc_info=True)
            else:
                logger.warning("microtubule mask expected but was None. Skipping microtubule mask writing for this frame.")
        else:
            logger.debug("microtubule mask generation is disabled for this series.")

        # B.2 Write the seed mask frame (if enabled)
        if self.cfg.generate_seed_mask:
            if seed_mask_img is not None:
                try:
                    if self.seed_mask_tiff_writer:
                        self.seed_mask_tiff_writer.append_data(seed_mask_img)
                        logger.debug("Appended seed mask to TIFF writer.")

                    if self.seed_mask_mp4_writer:
                        # Create and write the colorized preview for the mask
                        mask_vis_float = label2rgb(seed_mask_img, bg_label=0)
                        mask_vis_uint8 = (mask_vis_float * 255).astype(np.uint8)
                        mask_vis_bgr = cv2.cvtColor(mask_vis_uint8, cv2.COLOR_RGB2BGR)
                        self.seed_mask_mp4_writer.write(mask_vis_bgr)
                        logger.debug("Appended colorized seed mask to MP4 writer.")
                except Exception as e:
                    logger.error(f"Error appending seed mask frame: {e}", exc_info=True)
            else:
                logger.warning("Seed mask expected but was None. Skipping seed mask writing for this frame.")
        else:
            logger.debug("Seed mask generation is disabled for this series.")

        logger.debug("Finished appending frame.")

    def close(self):
        """Closes all writer objects to finalize files."""
        logger.info("Closing all file writers...")

        writers_to_close = [
            (self.video_tiff_writer, "TIFF video writer"),
            (self.video_mp4_writer, "MP4 video writer"),
            (self.gif_writer, "GIF writer"),
            (self.microtubule_mask_tiff_writer, "TIFF microtubule mask writer"),
            (self.microtubule_mask_mp4_writer, "MP4 microtubule mask writer"),
            (self.seed_mask_tiff_writer, "TIFF seed mask writer"),
            (self.seed_mask_mp4_writer, "MP4 seed mask writer"),
        ]

        for writer, name in writers_to_close:
            if writer:
                try:
                    if isinstance(writer, cv2.VideoWriter):
                        writer.release()
                        logger.debug(f"Released {name}.")
                    else:  # Assuming imageio writer
                        writer.close()
                        logger.debug(f"Closed {name}.")
                except Exception as e:
                    logger.error(f"Error closing {name}: {e}", exc_info=True)
            # else:
            # logger.debug(f"{name} was not initialized or already closed.")

        logger.info("All relevant file writers have been processed.")
