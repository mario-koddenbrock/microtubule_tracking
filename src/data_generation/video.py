import os
from typing import List, Tuple, Optional

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from config.synthetic_data import SyntheticDataConfig
from data_generation import utils
from data_generation.spots import SpotGenerator
from data_generation.tubuli import Microtubule
from data_generation.utils import apply_random_spots
from data_generation.utils import build_motion_seeds
from file_io.utils import save_ground_truth
from plotting.plotting import mask_to_color


def render_frame(
        cfg: SyntheticDataConfig,
        mts: list[Microtubule],
        frame_idx: int,
        fixed_spot_generator: SpotGenerator,
        moving_spot_generator: SpotGenerator,
        return_mask: bool = False,
) -> Tuple[np.ndarray, List[dict], Optional[np.ndarray]]:

    # 1) Prepare background and optional mask
    frame = np.full(cfg.img_size, cfg.background_level, dtype=np.float32)
    mask = np.zeros(cfg.img_size, dtype=np.uint16) if return_mask else None

    vignette = utils.compute_vignette(cfg)
    decay = np.exp(-frame_idx / cfg.bleach_tau) if np.isfinite(cfg.bleach_tau) else 1.0
    jitter = np.random.normal(0, cfg.jitter_px, 2) if cfg.jitter_px > 0 else np.zeros(2)
    cfg._frame_idx = frame_idx

    all_gt = []

    # 2) For each microtubule, step its length and draw
    for mt in mts:
        # A) Step to match the length profile:
        mt.step_to_length(frame_idx)

        # B) Temporarily add “jitter” to the entire chain’s base point:
        mt.base_point += jitter

        # C) Draw its wagons and collect ground truth:
        gt_info = mt.draw(frame, mask, cfg)
        all_gt.extend(gt_info)

        # D) Remove the jitter offset so it doesn’t accumulate next frame:
        mt.base_point -= jitter

    # Add background spots and noise after microtubules are drawn
    # 1. Apply fixed spots
    frame = fixed_spot_generator.apply(frame)

    # 2. Apply moving spots (at their current positions)
    frame = moving_spot_generator.apply(frame)

    # 3. Apply random spots (which are new every frame)
    frame = apply_random_spots(frame, cfg.random_spots)

    # 4. Update the state of the moving spots for the *next* frame
    moving_spot_generator.update()

    frame *= decay
    frame *= vignette
    frame *= decay
    frame *= vignette
    if cfg.gaussian_noise > 0.0:
        frame += np.random.normal(0, cfg.gaussian_noise, frame.shape).astype(np.float32)


    frame = utils.apply_global_blur(frame, cfg)

    frame = utils.annotate_frame(frame, frame_idx, fps=cfg.fps, show_time=cfg.show_time, show_scale=cfg.show_scale,
                                 scale_um_per_pixel=cfg.um_per_pixel, scale_length_um=cfg.scale_bar_um)

    if cfg.invert_contrast:
        frame = 2 * cfg.background_level - frame

    frame = np.clip(frame, 0.0, 1.0)
    frame_uint8 = (frame * 255).astype(np.uint8)

    return (frame_uint8, all_gt, mask) if return_mask else (frame_uint8, all_gt, None)


def generate_frames(cfg: SyntheticDataConfig, *, return_mask: bool = False):
    # 1) Build a list of Microtubule objects instead of raw “seeds”:
    mts = []

    for idx, (start_pt, motion_profile) in enumerate(build_motion_seeds(cfg), start=1):
        # 1) Randomize base orientation as before:
        base_orient = np.random.uniform(0.0, 2 * np.pi)

        # 2) Draw a per‐microtubule angle_change_prob ∈ [0, cfg.max_angle_change_prob]:
        angle_change_prob = np.random.uniform(0.0, cfg.max_angle_change_prob)

        # 3) Draw base-wagon length:
        base_len = np.random.uniform(
            cfg.min_base_wagon_length,
            cfg.max_base_wagon_length
        )

        # 4) Draw per-wagon length bounds:
        min_wagon_length = np.random.uniform(
            cfg.min_wagon_length_min,
            cfg.min_wagon_length_max
        )
        max_wagon_length = np.random.uniform(
            cfg.max_wagon_length_min,
            cfg.max_wagon_length_max
        )

        # 5) Instantiate Microtubule with its own angle_change_prob:
        mt = Microtubule(
            base_point=start_pt,
            base_orientation=base_orient,
            base_wagon_length=base_len,
            profile=motion_profile,
            max_num_wagons=cfg.max_num_wagons,
            max_angle=cfg.max_angle,
            angle_change_prob=angle_change_prob,
            min_wagon_length=min_wagon_length,
            max_wagon_length=max_wagon_length,
            instance_id=idx,
        )
        mt.instance_id = idx
        mts.append(mt)

    fixed_spot_generator = SpotGenerator(cfg.fixed_spots, cfg.img_size)
    moving_spot_generator = SpotGenerator(cfg.moving_spots, cfg.img_size)

    # 2) For each frame, step each microtubule and draw it:
    for frame_idx in range(cfg.num_frames):
        frame, all_gt, mask = render_frame(cfg, mts, frame_idx, fixed_spot_generator, moving_spot_generator,
                                           return_mask=return_mask)
        yield frame, all_gt, mask


def generate_video(cfg: SyntheticDataConfig, base_output_dir: str):
    """
    Generates a synthetic video sequence of microtubules along with ground truth annotations.

    Saves:
    - Video (MP4)
    - Animated preview (GIF)
    - Ground truth data (JSON)
    - Optional instance segmentation mask video (MP4)

    Returns:
        Tuple of file paths: (video, ground truth JSON, mask video or None)
    """
    os.makedirs(base_output_dir, exist_ok=True)
    video_path = os.path.join(base_output_dir, f"series_{cfg.id}.mp4")
    gif_path = os.path.join(base_output_dir, f"series_{cfg.id}.gif")
    mask_video_path = (os.path.join(base_output_dir, f"series_{cfg.id}_mask.mp4") if cfg.generate_mask else None)
    gt_path_json = os.path.join(base_output_dir, f"series_{cfg.id}_gt.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, cfg.fps, cfg.img_size[::-1])
    mask_writer = (cv2.VideoWriter(mask_video_path, fourcc, cfg.fps, cfg.img_size[::-1]) if cfg.generate_mask else None)

    frames = []
    mask_frames = []
    cfg._bend_params = {}  # Reset per-video bending memory

    for frame, gt_frame, mask in tqdm(generate_frames(cfg, return_mask=cfg.generate_mask), total=cfg.num_frames,
                                      desc=f"Series {cfg.id}"):
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask_frames.extend(gt_frame)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

        if cfg.generate_mask:
            mask_vis = mask_to_color(mask)
            mask_writer.write(mask_vis)

    writer.release()
    if cfg.generate_mask:
        mask_writer.release()

    imageio.mimsave(gif_path, frames, fps=cfg.fps)
    save_ground_truth(mask_frames, gt_path_json)

    return video_path, gt_path_json, mask_video_path


