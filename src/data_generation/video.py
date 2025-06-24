import os
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from config.synthetic_data import SyntheticDataConfig
from data_generation import utils
from data_generation.spots import SpotGenerator
from data_generation.tubuli import Microtubule
from data_generation.utils import apply_random_spots
from data_generation.utils import build_motion_seeds
from file_io.utils import save_ground_truth


# In file: data_generation/video.py

from typing import List, Tuple, Optional
import numpy as np

from config.synthetic_data import SyntheticDataConfig
from data_generation import utils
from data_generation.spots import SpotGenerator
from data_generation.tubuli import Microtubule
from file_io.writers import VideoOutputManager


def render_frame(
        cfg: SyntheticDataConfig,
        mts: list[Microtubule],
        frame_idx: int,
        fixed_spot_generator: SpotGenerator,
        moving_spot_generator: SpotGenerator,
        return_mask: bool = False,
) -> Tuple[np.ndarray, List[dict], Optional[np.ndarray]]:
    """
    Renders a single, complete frame of the synthetic video.

    This function follows a physically-motivated rendering pipeline:
    1.  Initializes a blank RGB canvas.
    2.  Renders the primary objects (microtubules) with their specific visual properties.
    3.  Adds secondary objects (spots).
    4.  Applies global optical and camera effects like bleaching, vignetting, and noise.
    5.  Adds final annotations and converts the frame to the output format.
    """
    # ─── 1. Initialization ──────────────────────────────────────────
    # The frame is 3-channel (RGB) float32 to accommodate color and precise calculations.
    frame = np.full((*cfg.img_size, 3), cfg.background_level, dtype=np.float32)
    mask = np.zeros(cfg.img_size, dtype=np.uint16) if return_mask else None
    gt_data = []

    # A single jitter offset is calculated for the entire frame.
    jitter = np.random.normal(0, cfg.jitter_px, 2) if cfg.jitter_px > 0 else np.zeros(2)
    # Pass the current frame index to the config for use in downstream functions.
    cfg._frame_idx = frame_idx


    # ─── 2. Simulate and Draw Microtubules ──────────────────────────
    for mt in mts:
        # A) Update the microtubule's length and dynamic state ("growing"/"shrinking")
        mt.step_to_length(frame_idx)

        # B) Apply the frame-wide jitter to the microtubule's anchor point
        mt.base_point += jitter

        # C) Draw the microtubule onto the frame and mask. The `draw` method handles
        #    the seed/dynamic coloring and the brighter +TIPs internally.
        gt_info = mt.draw(frame, mask, cfg)
        gt_data.extend(gt_info)

        # D) CRUCIAL: Remove the jitter so it doesn't accumulate on the next frame.
        mt.base_point -= jitter


    # ─── 3. Add Ancillary Objects (Spots) ───────────────────────────
    # Spots are currently rendered as white (added to all channels).
    frame = fixed_spot_generator.apply(frame)
    frame = moving_spot_generator.apply(frame)
    frame = utils.apply_random_spots(frame, cfg.random_spots)

    # Update the state of moving spots for the *next* frame.
    moving_spot_generator.update()


    # ─── 4. Apply Photophysics and Camera Effects ───────────────────
    # The order of these operations is important for physical realism.
    vignette = utils.compute_vignette(cfg)
    decay = np.exp(-frame_idx / cfg.bleach_tau) if np.isfinite(cfg.bleach_tau) else 1.0

    # 4a. Photobleaching: The overall signal fades over time.
    frame *= decay
    # 4b. Vignetting: The edges of the field of view are darker.
    #      (vignette is 2D, so we add a new axis to broadcast it across the 3 color channels)
    frame *= vignette[..., np.newaxis]

    # 4c. Mixed Noise Model:
    #     i) Photon Shot Noise (Poisson): Signal-dependent noise.
    if cfg.quantum_efficiency > 0:
        frame[frame < 0] = 0  # Ensure non-negative signal for Poisson distribution
        frame = np.random.poisson(frame * cfg.quantum_efficiency) / cfg.quantum_efficiency
    #     ii) Camera Read Noise (Gaussian): Signal-independent noise.
    if cfg.gaussian_noise > 0.0:
        frame += np.random.normal(0, cfg.gaussian_noise, frame.shape).astype(np.float32)

    # 4d. Global Blur: Simulates out-of-focus light and optical limitations.
    #     This is applied to the final, noisy image.
    frame = utils.apply_global_blur(frame, cfg)


    # ─── 5. Finalization and Formatting ─────────────────────────────
    # 5a. Annotations: Overlaid on top of the final image.
    frame = utils.annotate_frame(frame, frame_idx, fps=cfg.fps, show_time=cfg.show_time,
                                 show_scale=cfg.show_scale, scale_um_per_pixel=cfg.um_per_pixel,
                                 scale_length_um=cfg.scale_bar_um)

    # 5b. Contrast Inversion: (e.g., for dark-field style images)
    if cfg.invert_contrast:
        frame = 2 * cfg.background_level - frame

    # 5c. Clipping and Type Conversion: Convert the float image to a standard uint8 image.
    #     Clipping ensures that noise or other effects haven't pushed values outside the [0,1] range.
    frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)

    return (frame_uint8, gt_data, mask) if return_mask else (frame_uint8, gt_data, None)

def generate_frames(cfg: SyntheticDataConfig, *, return_mask: bool = False):
    # 1) Build a list of Microtubule objects
    mts = []
    # CHANGED: build_motion_seeds now just gives start points
    start_points = build_motion_seeds(cfg)

    for idx, start_pt in enumerate(start_points, start=1):
        base_orient = np.random.uniform(0.0, 2 * np.pi)
        angle_change_prob = np.random.uniform(0.0, cfg.max_angle_change_prob)
        base_len = np.random.uniform(
            cfg.min_base_wagon_length,
            cfg.max_base_wagon_length
        )

        # CHANGED: Instantiate Microtubule with the main config object
        # It will generate its own profile internally.
        mt = Microtubule(
            cfg=cfg,  # Pass the whole config
            base_point=start_pt,
            base_orientation=base_orient,
            base_wagon_length=base_len,
            instance_id=idx,
        )
        mts.append(mt)

    fixed_spot_generator = SpotGenerator(cfg.fixed_spots, cfg.img_size)
    moving_spot_generator = SpotGenerator(cfg.moving_spots, cfg.img_size)

    # 2) For each frame, step each microtubule and draw it:
    for frame_idx in range(cfg.num_frames):
        frame, gt_data, mask = render_frame(cfg, mts, frame_idx, fixed_spot_generator, moving_spot_generator,
                                            return_mask=return_mask)
        yield frame, gt_data, mask


def generate_video(cfg: SyntheticDataConfig, base_output_dir: str):
    """
    Generates a synthetic video sequence using a dedicated manager for file I/O.
    """
    # 1. Initialize the manager. It handles all file setup.
    output_manager = VideoOutputManager(cfg, base_output_dir)
    gt_json_path = os.path.join(base_output_dir, f"series_{cfg.id}_gt.json")

    try:
        all_gt_data = []
        cfg._bend_params = {}
        print(f"Generating and writing {cfg.num_frames} frames for Series {cfg.id}...")

        # 2. Process and write each frame one-by-one
        for frame_img_rgb, gt_data_for_frame, mask_img in tqdm(
                generate_frames(cfg, return_mask=cfg.generate_mask),
                total=cfg.num_frames
        ):
            # A. Accumulate ground truth data
            all_gt_data.extend(gt_data_for_frame)
            # B. Append frame and mask to all outputs via the manager
            output_manager.append(frame_img_rgb, mask_img)

        # 3. Save the collected ground truth data after the loop
        save_ground_truth(all_gt_data, gt_json_path)

    finally:
        # 4. Close all writers to finalize the files
        output_manager.close()

    # The paths are now internal to the manager, so we reconstruct them for the return statement
    video_tiff_path = os.path.join(base_output_dir, f"series_{cfg.id}_video.tif")
    masks_tiff_path = os.path.join(base_output_dir, f"series_{cfg.id}_masks.tif")
    return video_tiff_path, gt_json_path, masks_tiff_path