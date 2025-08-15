import concurrent.futures
import logging
import os
import random
from typing import List, Tuple, Optional, Dict, Any

import albumentations as A
import numpy as np
from tqdm import tqdm

from mt.config.synthetic_data import SyntheticDataConfig
from mt.data_generation import utils
from mt.data_generation.microtubule import Microtubule
from mt.data_generation.spots import SpotGenerator
from mt.file_io.writers import OutputManager

logger = logging.getLogger(f"mt.{__name__}")


def draw_mt(mt, cfg, frame, mt_mask, seed_mask, frame_idx, return_seed_mask, jitter):
    mt.step(cfg)
    mt.base_point += jitter
    local_frame = np.zeros_like(frame)
    local_mt_mask = np.zeros_like(mt_mask) if mt_mask is not None else None
    local_seed_mask = np.zeros_like(seed_mask) if seed_mask is not None else None
    gt_info = mt.draw(
        frame=local_frame,
        mt_mask=local_mt_mask,
        cfg=cfg,
        seed_mask=(local_seed_mask if frame_idx == 0 and return_seed_mask else None)
    )
    mt.base_point -= jitter
    return local_frame, local_mt_mask, local_seed_mask, gt_info


def render_frame(
        cfg: SyntheticDataConfig,
        mts: List[Microtubule],
        frame_idx: int,
        fixed_spot_generator: SpotGenerator,
        moving_spot_generator: SpotGenerator,
        aug_pipeline: Optional[A.Compose] = None,
        return_mt_mask: bool = False,
        return_seed_mask: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]:  # Adjusted return type hints
    """
    Renders a single frame of the synthetic video, including microtubules, spots,
    photophysics, camera effects, and augmentations.

    Returns:
        Tuple[np.ndarray, List[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
            - frame: The rendered image (uint8 RGB).
            - gt_data: List of ground truth dictionaries for this frame.
            - mt_mask: Optional mask for microtubules.
            - seed_mask: Optional mask for microtubule seeds (only for frame 0).
    """
    logger.debug(f"Rendering frame {frame_idx} for series ID {cfg.id}...")

    # ─── Initialization ──────────────────────────────────────────
    # Initialize background as a float32 array in the range [0, 1]
    frame = np.full((*cfg.img_size, 3), cfg.background_level, dtype=np.float32)
    mt_mask = np.zeros(cfg.img_size, dtype=np.uint16) if return_mt_mask else None

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

    # ─── Simulate and Draw Microtubules (Parallelized) ──────────
    args = [(mt, cfg, frame, mt_mask, seed_mask, frame_idx, return_seed_mask, jitter) for mt in mts]
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(draw_mt, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())


    # Sum all microtubule results into main frame/mask
    for local_frame, local_mt_mask, local_seed_mask, gt_info in results:
        frame += local_frame
        if mt_mask is not None and local_mt_mask is not None:
            mt_mask += local_mt_mask
        if seed_mask is not None and local_seed_mask is not None:
            seed_mask += local_seed_mask
        gt_data.extend(gt_info)

    # ─── Add Ancillary Objects (Spots) ───────────────────────────
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
            augmented = aug_pipeline(image=frame, mask=mt_mask if mt_mask is not None else None)
            frame = augmented['image']
            if mt_mask is not None:
                mt_mask = augmented['mask']
            logger.debug(f"Frame {frame_idx}: Albumentations applied.")

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
    frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    logger.debug(f"Frame {frame_idx}: Converted to uint8 (clipping applied).")

    # Add frame index to the ground truth data.
    for entry in gt_data:
        entry["frame_index"] = frame_idx
    logger.debug(f"Frame {frame_idx}: Ground truth data updated with frame_index. Total segments: {len(gt_data)}.")

    final_mt_mask = mt_mask if return_mt_mask else None
    final_seed_mask = seed_mask if return_seed_mask else None

    logger.debug(f"Frame {frame_idx} rendering complete.")
    return frame_uint8, gt_data, final_mt_mask, final_seed_mask


def generate_frames(
        cfg: SyntheticDataConfig, num_frames: int, return_mt_mask: bool = False, return_seed_mask: bool = False
):
    """
    Generates a sequence of synthetic video frames and associated data.
    This is a generator function, yielding one frame's data at a time.
    """
    logger.debug(f"Preparing to generate {num_frames} frames for series ID {cfg.id}...")

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
        logger.debug(f"Initialized {len(mts)} microtubules.")
    except Exception as e:
        logger.critical(f"Failed to initialize microtubules: {e}", exc_info=True)
        # If MTs cannot be initialized, we cannot generate frames. Raise or return empty.
        raise

    try:
        fixed_spot_generator = SpotGenerator(cfg.fixed_spots, cfg.img_size)
        logger.debug(f"Initialized fixed spot generator with {cfg.fixed_spots.count} spots.")
        moving_spot_generator = SpotGenerator(cfg.moving_spots, cfg.img_size)
        logger.debug(f"Initialized moving spot generator with {cfg.moving_spots.count} spots.")
    except Exception as e:
        logger.critical(f"Failed to initialize spot generators: {e}", exc_info=True)
        raise  # Critical setup failure

    aug_pipeline: Optional[A.Compose] = None
    try:
        if cfg.albumentations:
            aug_pipeline = utils.build_albumentations_pipeline(cfg.albumentations)
            # if aug_pipeline:
            #     logger.debug(f"Albumentations pipeline built successfully (master prob: {cfg.albumentations.p:.2f}).")
            # else:
            #     logger.debug("Albumentations config provided but pipeline is None (e.g., p=0 or no transforms).")
        else:
            logger.debug("Albumentations configuration is None. No augmentation pipeline will be built.")
    except Exception as e:
        logger.error(f"Error building Albumentations pipeline: {e}. Augmentations will be skipped.", exc_info=True)
        aug_pipeline = None  # Ensure it's None if building fails

    # For each frame, step each microtubule and draw it:
    for frame_idx in range(num_frames):
        try:
            frame, gt_data, mt_mask, seed_mask = render_frame(
                cfg=cfg,
                mts=mts,
                frame_idx=frame_idx,
                fixed_spot_generator=fixed_spot_generator,
                moving_spot_generator=moving_spot_generator,
                aug_pipeline=aug_pipeline,
                return_mt_mask=return_mt_mask,
                return_seed_mask=return_seed_mask,
            )
            logger.debug(f"Frame {frame_idx} yielded.")
            yield frame, gt_data, mt_mask, seed_mask, frame_idx
        except Exception as e:
            logger.error(f"Error rendering frame {frame_idx}: {e}. Skipping this frame.", exc_info=True)

def generate_video(
        cfg: SyntheticDataConfig,
        base_output_dir: str,
        num_png_frames: int = 10,
        is_for_expert_validation: bool = True,
) -> List[np.ndarray]:
    """
    Generates a full synthetic video, saves it, and optionally exports ground truth data.
    """
    logger.debug(f"Generating and writing {cfg.num_frames} frames for Series {cfg.id} into '{base_output_dir}'...")

    output_manager_main = OutputManager(cfg, os.path.join(base_output_dir, "full"))

    if is_for_expert_validation:
        output_manager_validation_set = OutputManager(cfg, os.path.join(base_output_dir, "validation"))
        validation_image_idx = random.randint(0, cfg.num_frames - 1)
    else:
        output_manager_validation_set = None
        validation_image_idx = -1  # No validation image for non-expert validation

    export_all_frames = (0 == num_png_frames) or (num_png_frames >= cfg.num_frames)
    export_current = True
    frame_idx_export = []

    if not export_all_frames:
        frame_idx_export = random.sample(range(cfg.num_frames), num_png_frames)

    frames: List[np.ndarray] = []
    frame_generator = generate_frames(cfg, cfg.num_frames,
                                      return_mt_mask=cfg.generate_mt_mask,
                                      return_seed_mask=cfg.generate_seed_mask)

    for frame_rgb, frame_gt, mt_mask, mt_seed_mask, frame_idx in tqdm(frame_generator,
            total=cfg.num_frames, desc=f"Series {cfg.id} frames"):

        frames.append(frame_rgb)
        if not export_all_frames:
            export_current = frame_idx in frame_idx_export
        output_manager_main.append(frame_idx, frame_rgb, mt_mask, mt_seed_mask, frame_gt, export_current)

        if is_for_expert_validation:
            export_png = (frame_idx == validation_image_idx)
            output_manager_validation_set.append(frame_idx, frame_rgb, mt_mask, mt_seed_mask, frame_gt, export_png)

    if output_manager_main:
        output_manager_main.close()

    if output_manager_validation_set:
        output_manager_validation_set.close()

    return frames
