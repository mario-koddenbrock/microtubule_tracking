import os
from typing import List, Tuple, Optional, Generator

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from config.synthetic_data import SyntheticDataConfig
from . import utils
from file_io.utils import save_ground_truth
from plotting.plotting import mask_to_color


def draw_instance(cfg, frame, mask, inst_id, slope, intercept, start_pt, end_pt, vignette, jitter, return_mask):
    """
    Draw a single microtubule instance on the frame with optional bending and jitter.

    Each instance is rendered as a series of 2D Gaussian spots aligned along the path between start_pt and end_pt.
    If bending is enabled, the path will deviate sinusoidally.

    Parameters:
        cfg: Configuration object with simulation parameters.
        frame: Image frame to draw on.
        mask: Optional segmentation mask to be updated.
        inst_id: Unique identifier for this instance.
        slope, intercept: Line parameters.
        start_pt, end_pt: Start and end coordinates of the instance.
        vignette: Vignetting mask applied to the image.
        jitter: Random offset applied to all positions.
        return_mask: Whether to update the instance mask.

    Returns:
        List of ground truth dictionaries for the instance.
    """
    gt_info = []
    start_pt_j = start_pt + jitter
    end_pt_j = end_pt + jitter

    vec = end_pt_j - start_pt_j
    length = np.linalg.norm(vec)
    if length == 0:
        return gt_info

    direction = vec / length
    perp = np.array([-direction[1], direction[0]])  # perpendicular vector
    step = 0.5  # step size along the microtubule
    n_steps = int(length / step) + 1

    for i in range(n_steps + 1):
        t = i / n_steps
        core_pos = start_pt_j + vec * t

        # Apply bending only in the second half and if parameters are available
        if t <= cfg.bend_straight_fraction or not hasattr(cfg, '_bend_params') or inst_id not in cfg._bend_params:
            pos = core_pos
        else:
            this_amp, dynamic_straight_fraction = cfg._bend_params[inst_id]
            if t <= (dynamic_straight_fraction + 1e-3): # avoid division by zero
                pos = core_pos
            else:
                # Apply sinusoidal deviation for bending
                step = np.pi * (t - dynamic_straight_fraction) / (1 - dynamic_straight_fraction)
                bend_offset = this_amp * np.sin(step) * perp
                pos = core_pos + bend_offset

        # Introduce variability in spot width
        local_sigma_x = cfg.sigma_x * (1 + np.random.normal(0, cfg.width_var_std))
        local_sigma_y = cfg.sigma_y * (1 + np.random.normal(0, cfg.width_var_std))

        # Draw only if position is inside bounds
        if 0 <= pos[0] < cfg.img_size[1] and 0 <= pos[1] < cfg.img_size[0]:
            ix = min(cfg.img_size[1] - 1, max(0, int(round(pos[0]))))
            iy = min(cfg.img_size[0] - 1, max(0, int(round(pos[1]))))
            utils.draw_tubulus(frame, pos, local_sigma_x, local_sigma_y, cfg.tubulus_contrast)
            if return_mask:
                mask[iy, ix] = inst_id

    gt_info.append({
        "frame_idx": cfg.frame_idx,
        "start": start_pt_j.tolist(),
        "end": end_pt_j.tolist(),
        "slope": slope,
        "intercept": intercept,
        "length": float(np.linalg.norm(end_pt_j - start_pt_j)),
        "instance_id": inst_id,
    })
    return gt_info


def render_frame(cfg: SyntheticDataConfig, seeds, frame_idx: int, *, return_mask: bool = False) -> Tuple[
    np.ndarray, List[dict], Optional[np.ndarray]]:
    """
    Renders a single frame with simulated microtubule instances.

    This includes:
    - Motion based on sawtooth profiles
    - Optional bending of microtubules
    - Vignetting, bleaching, noise, and optional mask rendering

    Returns:
        Tuple (frame_uint8, ground_truth, mask)
    """
    frame = np.full(cfg.img_size, cfg.background_level, dtype=np.float32)
    mask = np.zeros(cfg.img_size, dtype=np.uint16) if return_mask else None

    vignette = utils.compute_vignette(cfg)
    decay = np.exp(-frame_idx / cfg.bleach_tau) if np.isfinite(cfg.bleach_tau) else 1.0
    jitter = np.random.normal(0, cfg.jitter_px, 2) if cfg.jitter_px > 0 else np.zeros(2)
    cfg.frame_idx = frame_idx
    gt_frame = []
    rng = np.random.default_rng()

    for inst_id, ((slope, intercept), start_pt), motion_profile in ((idx + 1, *seed) for idx, seed in enumerate(seeds)):
        end_pt = utils.grow_shrink_seed(frame_idx, start_pt, slope, motion_profile, cfg.img_size, cfg.margin)
        this_amp, dynamic_straight_fraction, apply_bend = utils.update_bend_params(cfg, inst_id, motion_profile, start_pt,
                                                                             end_pt, rng)
        cfg._bend_params[inst_id] = (this_amp, dynamic_straight_fraction)
        gt_frame.extend(
            draw_instance(cfg, frame, mask, inst_id, slope, intercept, start_pt, end_pt, vignette, jitter, return_mask))

    # Add background spots and noise after microtubules are drawn
    frame = utils.add_fixed_spots(frame, cfg)
    frame = utils.add_moving_spots(frame, cfg)
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

    return (frame_uint8, gt_frame, mask) if return_mask else (frame_uint8, gt_frame, None)


def generate_frames(cfg: SyntheticDataConfig, *, return_mask: bool = False) -> Generator[
    Tuple[np.ndarray, List[dict], Optional[np.ndarray]], None, None]:
    """
    Generator that yields each frame of the video along with its ground truth and optional instance mask.

    Seeds (initial positions and motion profiles) are generated once and used for all frames.
    """
    seeds = utils.build_motion_seeds(cfg)
    for frame_idx in range(cfg.num_frames):
        yield render_frame(cfg, seeds, frame_idx, return_mask=return_mask)


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


if __name__ == "__main__":
    output_dir = "../data/synthetic"
    config_path = "../config/best_synthetic_config.json"

    config = SyntheticDataConfig.load()
    config.id = 29
    config.to_json(config_path)

    video_path, gt_path_json, gt_path_video = generate_video(config, output_dir)

    print(f"Saved video: {video_path}")
    print(f"Saved gt JSON: {gt_path_json}")
    print(f"Saved gt Video: {gt_path_video}")
