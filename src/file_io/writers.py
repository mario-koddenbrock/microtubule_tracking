import json
import logging
import os
from typing import Optional, Any

import cv2
import imageio
import numpy as np
from skimage.color import label2rgb

from config.synthetic_data import SyntheticDataConfig
from file_io.utils import CustomJsonEncoder

logger = logging.getLogger(f"mt.{__name__}")


class OutputManager:
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
        write_video_tiff: bool = True,
        write_masks_tiff: bool = True,
        write_video_mp4: bool = True,
        write_masks_mp4: bool = True,
        write_video_gif: bool = True,
        write_masks_gif: bool = True,
        write_video_pngs: bool = True,
        write_masks_pngs: bool = True,
        write_config: bool = True,
        write_gt: bool = True,
    ):
        """
        Initializes all file paths and writer objects based on the config.
        """
        logger.debug(f"Initializing OutputManager for series ID: {cfg.id}, output directory: '{base_output_dir}'")
        self.cfg = cfg
        self.base_output_dir = base_output_dir

        # Store output generation flags
        self.write_video_tiff = write_video_tiff
        self.write_masks_tiff = write_masks_tiff
        self.write_video_mp4 = write_video_mp4
        self.write_masks_mp4 = write_masks_mp4
        self.write_video_gif = write_video_gif
        self.write_masks_gif = write_masks_gif
        self.write_video_pngs = write_video_pngs
        self.write_masks_pngs = write_masks_pngs
        self.write_config = write_config
        self.write_gt = write_gt
        self.all_gt = []  # Store ground truth data if needed

        try:
            os.makedirs(base_output_dir, exist_ok=True)
            logger.debug(f"Ensured base output directory exists: {base_output_dir}")
        except OSError as e:
            logger.critical(f"Failed to create base output directory '{base_output_dir}': {e}", exc_info=True)
            raise

        self._initialize_paths()
        self._initialize_writers()
        logger.debug("OutputManager initialization complete.")

    def _initialize_paths(self):
        """Defines all output paths based on configuration."""
        base_name = f"series_{self.cfg.id}"
        self.paths = {}

        # Sequence file paths
        if self.write_video_tiff:
            self.paths['video_tiff'] = os.path.join(self.base_output_dir, "videos",
                                                    f"{base_name}_video.tif")
        if self.write_masks_tiff and self.cfg.generate_mt_mask:
            self.paths['mt_masks_tiff'] = os.path.join(self.base_output_dir, "video_masks",
                                                       f"{base_name}_masks.tif")

        # Preview MP4
        if self.write_video_mp4:
            self.paths['video_mp4'] = os.path.join(self.base_output_dir, "previews",
                                                   f"{base_name}_video_preview.mp4")
        if self.write_masks_mp4 and self.cfg.generate_mt_mask:
            self.paths['mt_masks_mp4'] = os.path.join(self.base_output_dir, "previews",
                                                      f"{base_name}_masks_preview.mp4")

        # Preview GIFs
        if self.write_video_gif:
            self.paths['video_gif'] = os.path.join(self.base_output_dir, "previews",
                                                   f"{base_name}_video_preview.gif")
        if self.write_masks_gif and self.cfg.generate_mt_mask:
            self.paths['mt_masks_gif'] = os.path.join(self.base_output_dir, "previews",
                                                      f"{base_name}_masks_preview.gif")

        # Seed masks
        if self.cfg.generate_seed_mask:
            if self.write_masks_tiff:
                self.paths['seed_masks_tiff'] = os.path.join(self.base_output_dir, "video_masks",
                                                             f"{base_name}_seed_masks.tif")
            if self.write_masks_mp4:
                self.paths['seed_masks_mp4'] = os.path.join(self.base_output_dir, "previews",
                                                            f"{base_name}_seed_masks_preview.mp4")
            if self.write_masks_gif:
                self.paths['seed_masks_gif'] = os.path.join(self.base_output_dir, "previews",
                                                            f"{base_name}_seed_masks_preview.gif")

        # Single-frame PNGs
        if self.write_video_pngs:
            self.paths['video_png_dir'] = os.path.join(self.base_output_dir, "images")
        if self.write_masks_pngs and self.cfg.generate_mt_mask:
            self.paths['mt_mask_png_dir'] = os.path.join(self.base_output_dir, "image_masks")
        if self.write_masks_pngs and self.cfg.generate_seed_mask:
            self.paths['seed_mask_png_dir'] = os.path.join(self.base_output_dir, "image_masks")

        # Config
        if self.write_config:
            self.paths['config_file'] = os.path.join(self.base_output_dir, "configs", f"{base_name}_config.json")
            try:
                self.cfg.save(self.paths['config_file'])
                logger.debug(f"Configuration saved to {self.paths['config_file']}")
            except Exception as e:
                logger.error(f"Failed to save configuration file: {e}", exc_info=True)

        # Ground truth file
        if self.write_gt:
            self.paths['ground_truth_file'] = os.path.join(self.base_output_dir, "gt", f"{base_name}_ground_truth.json")

        logger.debug(f"Output paths defined: {self.paths}")

        # Ensure all necessary directories exist
        for key, path in self.paths.items():
            if key.endswith('_dir'):
                try:
                    os.makedirs(path, exist_ok=True)
                    logger.debug(f"Ensured directory exists: {path}")
                except OSError as e:
                    logger.error(f"Failed to create directory '{path}': {e}", exc_info=True)
                    raise
            elif not os.path.exists(os.path.dirname(path)):
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    logger.debug(f"Created parent directory for file: {os.path.dirname(path)}")
                except OSError as e:
                    logger.error(f"Failed to create parent directory '{os.path.dirname(path)}': {e}", exc_info=True)
                    raise

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

        if self.cfg.generate_mt_mask:
            if self.write_masks_tiff:
                self.writers['mt_mask_tiff'] = _create_writer('tiff', 'mt_masks_tiff')
            if self.write_masks_mp4:
                self.writers['mt_mask_mp4'] = _create_writer('mp4', 'mt_masks_mp4')
            if self.write_masks_gif:
                self.writers['mt_mask_gif'] = _create_writer('gif', 'mt_masks_gif')

        if self.cfg.generate_seed_mask:
            if self.write_masks_tiff:
                self.writers['seed_mask_tiff'] = _create_writer('tiff', 'seed_masks_tiff')
            if self.write_masks_mp4:
                self.writers['seed_mask_mp4'] = _create_writer('mp4', 'seed_masks_mp4')
            if self.write_masks_gif:
                self.writers['seed_mask_gif'] = _create_writer('gif', 'seed_masks_gif')

    def append(
            self,
            frame_idx: int,
            frame_img_rgb: np.ndarray,
            mt_mask_img: Optional[np.ndarray],
            seed_mask_img: Optional[np.ndarray],
            frame_gt: Optional[list[dict[str, Any]]] = None,
            export_current_png: bool = True,
    ):
        logger.debug(f"Appending frame {frame_idx} to output writers...")
        frame_bgr = cv2.cvtColor(frame_img_rgb, cv2.COLOR_RGB2BGR)

        base_name = f"series_{self.cfg.id}"
        if self.cfg.num_frames > 1:
            frame_name = f"{base_name}_frame_{frame_idx:04d}.png"
        else:
            frame_name = f"{base_name}.png"

        # A. Write main video frame
        if self.writers.get('video_tiff'): self.writers['video_tiff'].append_data(frame_img_rgb)
        if self.writers.get('video_mp4'): self.writers['video_mp4'].write(frame_bgr)
        if self.writers.get('video_gif'): self.writers['video_gif'].append_data(frame_img_rgb)
        if self.write_video_pngs and export_current_png:
            path = os.path.join(self.paths['video_png_dir'], frame_name)
            cv2.imwrite(path, frame_bgr)

        # B. Write microtubule mask frame
        if self.cfg.generate_mt_mask and mt_mask_img is not None:
            if self.writers.get('mt_mask_tiff'): self.writers['mt_mask_tiff'].append_data(mt_mask_img)
            if self.write_masks_pngs and export_current_png:
                path = os.path.join(self.paths['mt_mask_png_dir'], frame_name)
                imageio.imwrite(path, mt_mask_img)
            if self.writers.get('mt_mask_mp4') or self.writers.get('mt_mask_gif'):
                mask_vis_rgb = (label2rgb(mt_mask_img, bg_label=0) * 255).astype(np.uint8)
                if self.writers.get('mt_mask_mp4'):
                    self.writers['mt_mask_mp4'].write(cv2.cvtColor(mask_vis_rgb, cv2.COLOR_RGB2BGR))
                if self.writers.get('mt_mask_gif'):
                    self.writers['mt_mask_gif'].append_data(mask_vis_rgb)

        # C. Write seed mask frame
        if self.cfg.generate_seed_mask and seed_mask_img is not None:
            if self.writers.get('seed_mask_tiff'): self.writers['seed_mask_tiff'].append_data(seed_mask_img)
            if self.write_masks_pngs and export_current_png:
                path = os.path.join(self.paths['seed_mask_png_dir'], frame_name)
                imageio.imwrite(path, seed_mask_img)
            if self.writers.get('seed_mask_mp4') or self.writers.get('seed_mask_gif'):
                mask_vis_rgb = (label2rgb(seed_mask_img, bg_label=0) * 255).astype(np.uint8)
                if self.writers.get('seed_mask_mp4'):
                    self.writers['seed_mask_mp4'].write(cv2.cvtColor(mask_vis_rgb, cv2.COLOR_RGB2BGR))
                if self.writers.get('seed_mask_gif'):
                    self.writers['seed_mask_gif'].append_data(mask_vis_rgb)

        # D. Write ground truth data if available
        if self.write_gt and frame_gt is not None:
            self.all_gt.append(frame_gt)

        logger.debug(f"Finished appending frame {frame_idx}.")

    def close(self):
        """Closes all writer objects to finalize files."""

        if self.write_gt:
            self.save_ground_truth()

        logger.debug("Closing all file writers...")
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
        logger.debug("All file writers have been processed.")

    def save_ground_truth(self):
        logger.debug(f"Attempting to save JSON ground truth to: {self.paths['ground_truth_file']}")
        try:
            output_dir = os.path.dirname(self.paths['ground_truth_file'])
            os.makedirs(output_dir, exist_ok=True)

            with open(self.paths['ground_truth_file'], "w") as fh:
                # Pass the custom encoder class to json.dump using the `cls` argument.
                json.dump(self.all_gt, fh, indent=2, cls=CustomJsonEncoder)
            logger.debug(f"Successfully saved JSON ground truth to: {self.paths['ground_truth_file']}")
        except Exception as e:
            logger.error(f"Failed to save JSON ground truth to {self.paths['ground_truth_file']}: {e}", exc_info=True)
            raise