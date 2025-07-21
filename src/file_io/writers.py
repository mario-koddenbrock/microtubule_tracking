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
    (TIFF, MP4, GIF, PNG) to simplify the main video generation loop. It allows
    for granular control over which output formats are generated.
    """

    def __init__(
        self,
        cfg: SyntheticDataConfig,
        base_output_dir: str,
        write_video_tiff: bool = False,
        write_masks_tiff: bool = False,
        write_video_mp4: bool = True,
        write_masks_mp4: bool = True,
        write_video_gif: bool = False,
        write_masks_gif: bool = False,
        write_video_pngs: bool = True,
        write_masks_pngs: bool = False,
    ):
        """
        Initializes all file paths and writer objects based on the config.
        """
        logger.info(f"Initializing VideoOutputManager for series ID: {cfg.id}, output directory: '{base_output_dir}'")
        self.cfg = cfg
        self.base_output_dir = base_output_dir
        self.frame_count = 0

        # Store output generation flags
        self.write_video_tiff = write_video_tiff
        self.write_masks_tiff = write_masks_tiff
        self.write_video_mp4 = write_video_mp4
        self.write_masks_mp4 = write_masks_mp4
        self.write_video_gif = write_video_gif
        self.write_masks_gif = write_masks_gif
        self.write_video_pngs = write_video_pngs
        self.write_masks_pngs = write_masks_pngs

        try:
            os.makedirs(base_output_dir, exist_ok=True)
            logger.debug(f"Ensured base output directory exists: {base_output_dir}")
        except OSError as e:
            logger.critical(f"Failed to create base output directory '{base_output_dir}': {e}", exc_info=True)
            raise

        self._initialize_paths()
        self._initialize_writers()
        logger.info("VideoOutputManager initialization complete.")

    def _initialize_paths(self):
        """Defines all output paths based on configuration."""
        base_name = f"series_{self.cfg.id}"
        self.paths = {}

        # Sequence file paths
        if self.write_video_tiff:
            self.paths['video_tiff'] = os.path.join(self.base_output_dir, f"{base_name}_video.tif")
        if self.write_masks_tiff and self.cfg.generate_microtubule_mask:
            self.paths['microtubule_masks_tiff'] = os.path.join(self.base_output_dir, f"{base_name}_masks.tif")
        if self.write_masks_tiff and self.cfg.generate_seed_mask:
            self.paths['seed_masks_tiff'] = os.path.join(self.base_output_dir, f"{base_name}_seed_masks.tif")
        if self.write_video_mp4:
            self.paths['video_mp4'] = os.path.join(self.base_output_dir, f"{base_name}_video_preview.mp4")
        if self.write_masks_mp4 and self.cfg.generate_microtubule_mask:
            self.paths['microtubule_masks_mp4'] = os.path.join(self.base_output_dir, f"{base_name}_masks_preview.mp4")
        if self.write_masks_mp4 and self.cfg.generate_seed_mask:
            self.paths['seed_masks_mp4'] = os.path.join(self.base_output_dir, f"{base_name}_seed_masks_preview.mp4")
        if self.write_video_gif:
            self.paths['video_gif'] = os.path.join(self.base_output_dir, f"{base_name}_video_preview.gif")
        if self.write_masks_gif and self.cfg.generate_microtubule_mask:
            self.paths['microtubule_masks_gif'] = os.path.join(self.base_output_dir, f"{base_name}_masks_preview.gif")
        if self.write_masks_gif and self.cfg.generate_seed_mask:
            self.paths['seed_masks_gif'] = os.path.join(self.base_output_dir, f"{base_name}_seed_masks_preview.gif")

        # Per-frame directory paths
        if self.write_video_pngs:
            self.paths['video_png_dir'] = os.path.join(self.base_output_dir, f"{base_name}_video_frames")
            os.makedirs(self.paths['video_png_dir'], exist_ok=True)
        if self.write_masks_pngs and self.cfg.generate_microtubule_mask:
            self.paths['microtubule_mask_png_dir'] = os.path.join(self.base_output_dir, f"{base_name}_mask_frames")
            os.makedirs(self.paths['microtubule_mask_png_dir'], exist_ok=True)
        if self.write_masks_pngs and self.cfg.generate_seed_mask:
            self.paths['seed_mask_png_dir'] = os.path.join(self.base_output_dir, f"{base_name}_seed_mask_frames")
            os.makedirs(self.paths['seed_mask_png_dir'], exist_ok=True)

        logger.debug(f"Output paths defined: {self.paths}")

    def _initialize_writers(self):
        """Initializes all writer objects."""
        self.writers = {}
        img_h, img_w = self.cfg.img_size
        mp4_fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        def _create_writer(writer_type, path_key, **kwargs):
            try:
                if writer_type == 'tiff':
                    return imageio.get_writer(self.paths[path_key], format="TIFF")
                elif writer_type == 'mp4':
                    writer = cv2.VideoWriter(self.paths[path_key], mp4_fourcc, self.cfg.fps, (img_w, img_h), **kwargs)
                    if not writer.isOpened():
                        raise IOError(f"OpenCV VideoWriter failed to open for {path_key}.")
                    return writer
                elif writer_type == 'gif':
                    return imageio.get_writer(self.paths[path_key], fps=self.cfg.fps, mode="I", loop=0)
            except Exception as e:
                logger.error(f"Failed to initialize {writer_type} writer for {path_key} at {self.paths.get(path_key)}: {e}", exc_info=True)
            return None

        if self.write_video_tiff:
            self.writers['video_tiff'] = _create_writer('tiff', 'video_tiff')
        if self.write_video_mp4:
            self.writers['video_mp4'] = _create_writer('mp4', 'video_mp4')
        if self.write_video_gif:
            self.writers['video_gif'] = _create_writer('gif', 'video_gif')

        if self.cfg.generate_microtubule_mask:
            if self.write_masks_tiff:
                self.writers['microtubule_mask_tiff'] = _create_writer('tiff', 'microtubule_masks_tiff')
            if self.write_masks_mp4:
                self.writers['microtubule_mask_mp4'] = _create_writer('mp4', 'microtubule_masks_mp4')
            if self.write_masks_gif:
                self.writers['microtubule_mask_gif'] = _create_writer('gif', 'microtubule_masks_gif')

        if self.cfg.generate_seed_mask:
            if self.write_masks_tiff:
                self.writers['seed_mask_tiff'] = _create_writer('tiff', 'seed_masks_tiff')
            if self.write_masks_mp4:
                self.writers['seed_mask_mp4'] = _create_writer('mp4', 'seed_masks_mp4')
            if self.write_masks_gif:
                self.writers['seed_mask_gif'] = _create_writer('gif', 'seed_masks_gif')

    def append(
        self, frame_img_rgb: np.ndarray, microtubule_mask_img: Optional[np.ndarray], seed_mask_img: Optional[np.ndarray]
    ):
        """Appends a new frame and its masks to all configured outputs."""
        logger.debug(f"Appending frame {self.frame_count} to output writers...")
        frame_bgr = cv2.cvtColor(frame_img_rgb, cv2.COLOR_RGB2BGR)

        # A. Write main video frame
        if self.writers.get('video_tiff'): self.writers['video_tiff'].append_data(frame_img_rgb)
        if self.writers.get('video_mp4'): self.writers['video_mp4'].write(frame_bgr)
        if self.writers.get('video_gif'): self.writers['video_gif'].append_data(frame_img_rgb)
        if self.write_video_pngs:
            path = os.path.join(self.paths['video_png_dir'], f"frame_{self.frame_count:04d}.png")
            cv2.imwrite(path, frame_bgr)

        # B. Write microtubule mask frame
        if self.cfg.generate_microtubule_mask and microtubule_mask_img is not None:
            if self.writers.get('microtubule_mask_tiff'): self.writers['microtubule_mask_tiff'].append_data(microtubule_mask_img)
            if self.write_masks_pngs:
                path = os.path.join(self.paths['microtubule_mask_png_dir'], f"mask_{self.frame_count:04d}.png")
                imageio.imwrite(path, microtubule_mask_img)
            if self.writers.get('microtubule_mask_mp4') or self.writers.get('microtubule_mask_gif'):
                mask_vis_rgb = (label2rgb(microtubule_mask_img, bg_label=0) * 255).astype(np.uint8)
                if self.writers.get('microtubule_mask_mp4'):
                    self.writers['microtubule_mask_mp4'].write(cv2.cvtColor(mask_vis_rgb, cv2.COLOR_RGB2BGR))
                if self.writers.get('microtubule_mask_gif'):
                    self.writers['microtubule_mask_gif'].append_data(mask_vis_rgb)

        # C. Write seed mask frame
        if self.cfg.generate_seed_mask and seed_mask_img is not None:
            if self.writers.get('seed_mask_tiff'): self.writers['seed_mask_tiff'].append_data(seed_mask_img)
            if self.write_masks_pngs:
                path = os.path.join(self.paths['seed_mask_png_dir'], f"seed_mask_{self.frame_count:04d}.png")
                imageio.imwrite(path, seed_mask_img)
            if self.writers.get('seed_mask_mp4') or self.writers.get('seed_mask_gif'):
                mask_vis_rgb = (label2rgb(seed_mask_img, bg_label=0) * 255).astype(np.uint8)
                if self.writers.get('seed_mask_mp4'):
                    self.writers['seed_mask_mp4'].write(cv2.cvtColor(mask_vis_rgb, cv2.COLOR_RGB2BGR))
                if self.writers.get('seed_mask_gif'):
                    self.writers['seed_mask_gif'].append_data(mask_vis_rgb)

        self.frame_count += 1
        logger.debug(f"Finished appending frame {self.frame_count - 1}.")

    def close(self):
        """Closes all writer objects to finalize files."""
        logger.info("Closing all file writers...")
        for name, writer in self.writers.items():
            if writer:
                try:
                    if isinstance(writer, cv2.VideoWriter):
                        writer.release()
                    else:  # imageio writer
                        writer.close()
                    logger.debug(f"Closed {name} writer.")
                except Exception as e:
                    logger.error(f"Error closing {name} writer: {e}", exc_info=True)
        logger.info("All file writers have been processed.")