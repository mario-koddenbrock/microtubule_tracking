import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config.synthetic_data import SyntheticDataConfig
from file_io.utils import save_ground_truth
from file_io.writers import VideoOutputManager
from . import utils
from .spots import SpotGenerator
from .tubuli import Microtubule
from .utils import build_motion_seeds


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
    """
    # ─── 1. Initialization ──────────────────────────────────────────
    # The frame is initialized to the background level. The drawing functions
    # will handle making objects brighter or darker from this baseline.
    frame = np.full((*cfg.img_size, 3), cfg.background_level, dtype=np.float32)
    mask = np.zeros(cfg.img_size, dtype=np.uint16) if return_mask else None
    gt_data = []

    jitter = np.random.normal(0, cfg.jitter_px, 2) if cfg.jitter_px > 0 else np.zeros(2)

    # ─── 2. Simulate and Draw Microtubules ──────────────────────────
    for mt in mts:
        mt.step_to_length(frame_idx)
        mt.base_point += jitter
        gt_info = mt.draw(frame, mask, cfg)
        gt_data.extend(gt_info)
        mt.base_point -= jitter


    # ─── 3. Add Ancillary Objects (Spots) ───────────────────────────
    frame = fixed_spot_generator.apply(frame)
    frame = moving_spot_generator.apply(frame)
    frame = SpotGenerator.apply_random_spots(frame, cfg.random_spots)
    moving_spot_generator.update()

    # ─── 4. Apply Photophysics and Camera Effects ───────────────────
    vignette = utils.compute_vignette(cfg)
    decay = np.exp(-frame_idx / cfg.bleach_tau) if np.isfinite(cfg.bleach_tau) else 1.0
    frame *= decay
    frame *= vignette[..., np.newaxis]

    
    if cfg.quantum_efficiency > 0:
        frame[frame < 0] = 0
        frame = np.random.poisson(frame * cfg.quantum_efficiency) / cfg.quantum_efficiency

    if cfg.gaussian_noise > 0.0:
        frame += np.random.normal(0, cfg.gaussian_noise, frame.shape).astype(np.float32)

    frame = utils.apply_global_blur(frame, cfg)

    # ─── 5. Finalization and Formatting ─────────────────────────────
    frame = utils.annotate_frame(frame, cfg, frame_idx)

    frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)

    # 6. Add frame index to the ground truth data.
    for entry in gt_data:
        entry['frame_index'] = frame_idx

    # plt.imshow(frame_uint8)
    # plt.axis('off')
    # plt.show()

    return (frame_uint8, gt_data, mask) if return_mask else (frame_uint8, gt_data, None)

def generate_frames(cfg: SyntheticDataConfig, *, return_mask: bool = False):
    # 1) Build a list of Microtubule objects
    mts = []
    start_points = build_motion_seeds(cfg)

    for idx, start_pt in enumerate(start_points, start=1):
        mts.append(
            Microtubule(
                cfg=cfg,
                base_point=start_pt,
                instance_id=idx,
            )
        )

    fixed_spot_generator = SpotGenerator(cfg.fixed_spots, cfg.img_size)
    moving_spot_generator = SpotGenerator(cfg.moving_spots, cfg.img_size)

    # 2) For each frame, step each microtubule and draw it:
    for frame_idx in range(cfg.num_frames):
        frame, gt_data, mask = render_frame(
            cfg, mts, frame_idx, fixed_spot_generator, moving_spot_generator, return_mask
        )
        yield frame, gt_data, mask


def generate_video(
        cfg: SyntheticDataConfig,
        base_output_dir: str,
        export_gt_data: bool = True,
):
    """
    Generates a synthetic video sequence using a dedicated manager for file I/O.
    """
    # 1. Initialize the manager. It handles all file setup.
    output_manager = VideoOutputManager(cfg, base_output_dir)
    gt_json_path = os.path.join(base_output_dir, f"series_{cfg.id}_gt.json")

    try:
        all_gt_data = []
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
        if export_gt_data:
            save_ground_truth(all_gt_data, gt_json_path)

    finally:
        # 4. Close all writers to finalize the files
        output_manager.close()

    # The paths are now internal to the manager, so we reconstruct them for the return statement
    video_tiff_path = os.path.join(base_output_dir, f"series_{cfg.id}_video.tif")
    masks_tiff_path = os.path.join(base_output_dir, f"series_{cfg.id}_masks.tif")
    return video_tiff_path, gt_json_path, masks_tiff_path