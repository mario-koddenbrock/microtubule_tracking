import logging
import os
from typing import List, Tuple, Optional, Dict, Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config.synthetic_data import SyntheticDataConfig
from data_generation import utils
from data_generation.microtubule import Microtubule
from data_generation.spots import SpotGenerator
from file_io.utils import save_ground_truth
from file_io.writers import VideoOutputManager
from plotting.plotting import show_frame

logger = logging.getLogger(f"mt.{__name__}")


def render_frame(
        cfg: SyntheticDataConfig,
        mts: List[Microtubule],
        frame_idx: int,
        fixed_spot_generator: SpotGenerator,
        moving_spot_generator: SpotGenerator,
        aug_pipeline: Optional[A.Compose] = None,
        return_microtubule_mask: bool = False,
        return_seed_mask: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]:  # Adjusted return type hints
    """
    Renders a single frame of the synthetic video, including microtubules, spots,
    photophysics, camera effects, and augmentations.

    Returns:
        Tuple[np.ndarray, List[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
            - frame: The rendered image (uint8 RGB).
            - gt_data: List of ground truth dictionaries for this frame.
            - microtubule_mask: Optional mask for microtubules.
            - seed_mask: Optional mask for microtubule seeds (only for frame 0).
    """
    logger.debug(f"Rendering frame {frame_idx} for series ID {cfg.id}...")

    # ─── Initialization ──────────────────────────────────────────
    # Initialize background as a float32 array in the range [0, 1]
    frame = np.full((*cfg.img_size, 3), cfg.background_level, dtype=np.float32)
    microtubule_mask = np.zeros(cfg.img_size, dtype=np.uint16) if return_microtubule_mask else None

    # seed_mask is only for the first frame and if requested
    seed_mask = None
    if frame_idx == 0 and return_seed_mask:
        seed_mask = np.zeros(cfg.img_size, dtype=np.uint16)
        logger.debug(f"Frame {frame_idx}: Initialized seed_mask.")

    gt_data: List[Dict[str, Any]] = []

    # Jitter is applied to microtubule base points
    jitter = np.random.normal(0, cfg.jitter_px, 2).astype(np.float32) if cfg.jitter_px > 0 else np.zeros(2,
                                                                                                         dtype=np.float32)
    if cfg.jitter_px > 0:
        logger.debug(f"Frame {frame_idx}: Applying jitter: {jitter.tolist()}")

    # ─── Simulate and Draw Microtubules ──────────────────────────
    logger.debug(f"Frame {frame_idx}: Simulating and drawing {len(mts)} microtubules.")
    for mt_idx, mt in enumerate(tqdm(mts, desc=f"Frame {frame_idx} MTs", unit="MT")):
        try:
            mt.step()
            mt.base_point += jitter

            # Pass seed_mask only if it's the first frame and requested
            gt_info = mt.draw(
                frame=frame,
                microtubule_mask=microtubule_mask,
                cfg=cfg,
                seed_mask=(seed_mask if frame_idx == 0 and return_seed_mask else None)
            )
            gt_data.extend(gt_info)
            mt.base_point -= jitter
            logger.debug(f"Frame {frame_idx}, MT {mt.instance_id}: Drawn with {len(gt_info)} segments.")
        except Exception as e:
            logger.error(f"Frame {frame_idx}, MT {mt.instance_id}: Error drawing microtubule: {e}", exc_info=True)

    # ─── Add Ancillary Objects (Spots) ───────────────────────────
    logger.debug(f"Frame {frame_idx}: Applying spots.")
    try:
        frame = fixed_spot_generator.apply(frame)
        logger.debug(f"Frame {frame_idx}: Fixed spots applied.")
        frame = moving_spot_generator.apply(frame)
        logger.debug(f"Frame {frame_idx}: Moving spots applied.")
        frame = SpotGenerator.apply_random_spots(frame, cfg.random_spots)
        logger.debug(f"Frame {frame_idx}: Random spots applied.")
        moving_spot_generator.update()  # Update moving spot positions for next frame
        logger.debug(f"Frame {frame_idx}: Moving spot generator updated for next frame.")
    except Exception as e:
        logger.error(f"Frame {frame_idx}: Error applying spots: {e}", exc_info=True)

    # ─── Apply Photophysics and Camera Effects ───────────────────
    logger.debug(f"Frame {frame_idx}: Applying photophysics and camera effects.")
    try:
        vignette = utils.compute_vignette(cfg)
        frame *= vignette[..., np.newaxis]
        logger.debug(f"Frame {frame_idx}: Applied vignetting.")

        # Apply red channel noise if specified
        if cfg.red_channel_noise_std > 0.0:
            red_noise = np.random.normal(0, cfg.red_channel_noise_std, frame.shape[:2]).astype(np.float32)
            frame[..., 0] += red_noise
            frame[..., 0] = np.clip(frame[..., 0], 0, 1)  # Ensure red channel stays in [0, 1]
            logger.debug(f"Frame {frame_idx}: Applied red channel noise (std={cfg.red_channel_noise_std:.4f}).")
        else:
            logger.debug(f"Frame {frame_idx}: Skipping red channel noise (std={cfg.red_channel_noise_std:.4f}).")

        if cfg.quantum_efficiency > 0:
            frame[frame < 0] = 0  # Clamp negative values before Poisson noise
            frame = np.random.poisson(frame * cfg.quantum_efficiency) / cfg.quantum_efficiency
            logger.debug(f"Frame {frame_idx}: Applied Poisson noise (QE={cfg.quantum_efficiency:.2f}).")
        else:
            logger.debug(f"Frame {frame_idx}: Skipping Poisson noise (QE={cfg.quantum_efficiency:.2f}).")

        if cfg.gaussian_noise > 0.0:
            frame += np.random.normal(0, cfg.gaussian_noise, frame.shape).astype(np.float32)
            logger.debug(f"Frame {frame_idx}: Applied Gaussian noise (std={cfg.gaussian_noise:.4f}).")
        else:
            logger.debug(f"Frame {frame_idx}: Skipping Gaussian noise (std={cfg.gaussian_noise:.4f}).")

        frame = utils.apply_global_blur(frame, cfg)
        logger.debug(f"Frame {frame_idx}: Applied global blur (sigma={cfg.global_blur_sigma:.2f}).")

        # show_frame(frame, title="Before Albumentations")

        if cfg.contrast > 0.0:
            frame = utils.apply_contrast(frame, cfg.contrast)
            logger.debug(f"Frame {frame_idx}: Applied contrast adjustment (factor={cfg.contrast:.2f}).")

        if cfg.brightness > 0.0:
            frame = utils.apply_brightness(frame, cfg.brightness)
            logger.debug(f"Frame {frame_idx}: Applied brightness adjustment (factor={cfg.brightness:.2f}).")

        # show_frame(frame, title="After Albumentations")

    except Exception as e:
        logger.error(f"Frame {frame_idx}: Error applying photophysics/camera effects: {e}", exc_info=True)

    # ─── Apply Augmentations ────────────────────────────────
    if aug_pipeline and cfg.albumentations and cfg.albumentations.p > 0:
        logger.debug(f"Frame {frame_idx}: Applying Albumentations (p={cfg.albumentations.p:.2f}).")
        try:
            # Albumentations expects uint8 or float, ensure float [0,1]
            # Convert back to original frame type if needed
            # For simplicity, passing float32 as is.
            plt.imshow(frame)
            plt.axis("off")
            plt.title(f"Frame {frame_idx}")
            plt.show()
            augmented = aug_pipeline(image=frame, mask=microtubule_mask if microtubule_mask is not None else None)
            frame = augmented['image']
            if microtubule_mask is not None:
                microtubule_mask = augmented['mask']
            logger.debug(f"Frame {frame_idx}: Albumentations applied.")

            plt.imshow(frame)
            plt.axis("off")
            plt.title(f"Augmented Frame {frame_idx}")
            plt.show()
        except Exception as e:
            logger.error(f"Frame {frame_idx}: Error applying Albumentations: {e}", exc_info=True)
            # Log error but don't stop rendering the frame
    elif aug_pipeline and cfg.albumentations and cfg.albumentations.p == 0:
        logger.debug(f"Frame {frame_idx}: Albumentations pipeline exists but master probability (p) is 0. Skipping.")
    else:
        logger.debug(f"Frame {frame_idx}: No Albumentations pipeline or config. Skipping.")

    # ─── Finalization and Formatting ─────────────────────────────
    logger.debug(f"Frame {frame_idx}: Finalizing frame and ground truth.")
    try:
        frame = utils.annotate_frame(frame, cfg, frame_idx)
        logger.debug(f"Frame {frame_idx}: Annotations applied.")
    except Exception as e:
        logger.error(f"Frame {frame_idx}: Error annotating frame: {e}", exc_info=True)

    # Convert to uint8, clipping values to [0,1] range first
    frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    logger.debug(f"Frame {frame_idx}: Converted to uint8 (clipping applied).")

    # Add frame index to the ground truth data.
    for entry in gt_data:
        entry["frame_index"] = frame_idx
    logger.debug(f"Frame {frame_idx}: Ground truth data updated with frame_index. Total segments: {len(gt_data)}.")

    final_microtubule_mask = microtubule_mask if return_microtubule_mask else None
    final_seed_mask = seed_mask if return_seed_mask else None

    logger.debug(f"Frame {frame_idx} rendering complete.")
    return frame_uint8, gt_data, final_microtubule_mask, final_seed_mask


def generate_frames(
        cfg: SyntheticDataConfig, num_frames: int, return_microtubule_mask: bool = False, return_seed_mask: bool = False
):
    """
    Generates a sequence of synthetic video frames and associated data.
    This is a generator function, yielding one frame's data at a time.
    """
    logger.info(f"Preparing to generate {num_frames} frames for series ID {cfg.id}...")

    mts: List[Microtubule] = []
    try:
        start_points = utils.build_motion_seeds(cfg)
        logger.debug(f"Built {len(start_points)} initial motion seeds for microtubules.")
        for idx, start_pt in enumerate(start_points, start=1):
            mts.append(
                Microtubule(
                    cfg=cfg,
                    base_point=start_pt,
                    instance_id=idx,
                )
            )
        logger.info(f"Initialized {len(mts)} microtubules.")
    except Exception as e:
        logger.critical(f"Failed to initialize microtubules: {e}", exc_info=True)
        # If MTs cannot be initialized, we cannot generate frames. Raise or return empty.
        raise

    try:
        fixed_spot_generator = SpotGenerator(cfg.fixed_spots, cfg.img_size)
        logger.info(f"Initialized fixed spot generator with {cfg.fixed_spots.count} spots.")
        moving_spot_generator = SpotGenerator(cfg.moving_spots, cfg.img_size)
        logger.info(f"Initialized moving spot generator with {cfg.moving_spots.count} spots.")
    except Exception as e:
        logger.critical(f"Failed to initialize spot generators: {e}", exc_info=True)
        raise  # Critical setup failure

    aug_pipeline: Optional[A.Compose] = None
    try:
        if cfg.albumentations:
            aug_pipeline = utils.build_albumentations_pipeline(cfg.albumentations)
            # if aug_pipeline:
            #     logger.info(f"Albumentations pipeline built successfully (master prob: {cfg.albumentations.p:.2f}).")
            # else:
            #     logger.info("Albumentations config provided but pipeline is None (e.g., p=0 or no transforms).")
        else:
            logger.info("Albumentations configuration is None. No augmentation pipeline will be built.")
    except Exception as e:
        logger.error(f"Error building Albumentations pipeline: {e}. Augmentations will be skipped.", exc_info=True)
        aug_pipeline = None  # Ensure it's None if building fails

    # For each frame, step each microtubule and draw it:
    for frame_idx in range(num_frames):
        try:
            frame, gt_data, microtubule_mask, seed_mask = render_frame(
                cfg=cfg,
                mts=mts,
                frame_idx=frame_idx,
                fixed_spot_generator=fixed_spot_generator,
                moving_spot_generator=moving_spot_generator,
                aug_pipeline=aug_pipeline,
                return_microtubule_mask=return_microtubule_mask,
                return_seed_mask=return_seed_mask,
            )
            logger.debug(f"Frame {frame_idx} yielded.")
            yield frame, gt_data, microtubule_mask, seed_mask
        except Exception as e:
            logger.error(f"Error rendering frame {frame_idx}: {e}. Skipping this frame.", exc_info=True)

def generate_video(
        cfg: SyntheticDataConfig,
        base_output_dir: str,
        export_gt_data: bool = True,
) -> List[np.ndarray]:
    """
    Generates a full synthetic video, saves it, and optionally exports ground truth data.
    """
    logger.info(f"Generating and writing {cfg.num_frames} frames for Series {cfg.id} into '{base_output_dir}'...")

    try:
        output_manager = VideoOutputManager(cfg, base_output_dir)
        logger.info(f"VideoOutputManager initialized for output directory: {base_output_dir}")
    except Exception as e:
        logger.critical(f"Failed to initialize VideoOutputManager for '{base_output_dir}': {e}", exc_info=True)
        raise  # Critical error, cannot save video

    gt_json_path = os.path.join(base_output_dir, f"series_{cfg.id}_gt.json")
    logger.debug(f"Ground truth JSON path: {gt_json_path}")

    all_gt_data: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []

    try:
        # Process and write each frame one-by-one using the generator
        for frame_img_rgb, gt_data_for_frame, microtubule_mask_img, seed_mask_img in (
                generate_frames(cfg, cfg.num_frames,
                                return_microtubule_mask=cfg.generate_microtubule_mask,
                                return_seed_mask=cfg.generate_seed_mask)):

            frames.append(frame_img_rgb)
            all_gt_data.extend(gt_data_for_frame)
            try:
                if output_manager:
                    output_manager.append(frame_img_rgb, microtubule_mask_img, seed_mask_img)
                    logger.debug(f"Appended frame to output manager.")
                else:
                    logger.warning("Output manager is None, skipping frame append.")
            except Exception as e:
                logger.error(f"Error appending frame to output manager: {e}. Continuing with next frame.",
                             exc_info=True)

        if export_gt_data:
            try:
                save_ground_truth(all_gt_data, gt_json_path)
                logger.info(f"Ground truth data saved to: {gt_json_path}")
            except Exception as e:
                logger.error(f"Error saving ground truth data to {gt_json_path}: {e}", exc_info=True)
        else:
            logger.info("Ground truth export disabled.")

    except Exception as e:
        logger.critical(f"A critical error occurred during video generation for Series {cfg.id}: {e}", exc_info=True)
        # Re-raise to signal a major failure to the calling process.
        raise

    finally:
        if output_manager:
            try:
                output_manager.close()
                logger.info(f"Video output manager closed for Series {cfg.id}.")
            except Exception as e:
                logger.error(f"Error closing video output manager: {e}", exc_info=True)
        else:
            logger.warning("Output manager was not initialized, nothing to close.")

    logger.info(f"Video generation for Series {cfg.id} completed.")
    return frames