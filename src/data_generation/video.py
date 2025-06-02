import os
from typing import List, Tuple, Optional, Generator

import cv2
import numpy as np
from tqdm import tqdm

from data_generation.config import SyntheticDataConfig
from data_generation.utils import (
    draw_tubulus,
    grow_shrink_seed, apply_global_blur, add_fixed_spots, add_moving_spots, annotate_frame,
)
from data_generation.utils import build_motion_seeds
from file_io.utils import save_ground_truth
from plotting.plotting import mask_to_color


def render_frame(
        cfg: SyntheticDataConfig,
        seeds,
        frame_idx: int,
        *,
        return_mask: bool = False,
) -> Tuple[np.ndarray, List[dict], Optional[np.ndarray]]:
    bg_level = cfg.background_level
    noise_std = cfg.gaussian_noise
    bleach_tau = cfg.bleach_tau
    jitter_px = cfg.jitter_px
    vignette_k = cfg.vignetting_strength

    invert_contrast = cfg.invert_contrast
    width_var_std = cfg.width_var_std
    bend_amplitude = cfg.bend_amplitude
    bend_prob = cfg.bend_prob
    bend_straight_fraction = cfg.bend_straight_fraction

    sigma_x = cfg.sigma_x
    sigma_y = cfg.sigma_y

    frame = np.full(cfg.img_size, bg_level, dtype=np.float32)
    mask: Optional[np.ndarray] = None
    if return_mask:
        mask = np.zeros(cfg.img_size, dtype=np.uint16)

    if vignette_k > 0.0:
        yy, xx = np.mgrid[:cfg.img_size[0], :cfg.img_size[1]]
        norm_x = (xx - cfg.img_size[1] / 2) / (cfg.img_size[1] / 2)
        norm_y = (yy - cfg.img_size[0] / 2) / (cfg.img_size[0] / 2)
        vignette = 1.0 - vignette_k * (norm_x ** 2 + norm_y ** 2)
        vignette = np.clip(vignette, 0.5, 1.0)
    else:
        vignette = 1.0

    decay = np.exp(-frame_idx / bleach_tau) if np.isfinite(bleach_tau) else 1.0
    jitter = np.random.normal(0, jitter_px, 2) if jitter_px > 0 else np.zeros(2)
    gt_frame: List[dict] = []
    rng = np.random.default_rng()

    for inst_id, ((slope, intercept), start_pt), motion_profile in ((idx + 1, *seed) for idx, seed in enumerate(seeds)):

        if not hasattr(cfg, "_bend_params"):
            cfg._bend_params = {}

        end_pt = grow_shrink_seed(frame_idx, start_pt, slope, motion_profile, cfg.img_size, cfg.margin)
        total_length = np.linalg.norm(end_pt - start_pt)

        if inst_id not in cfg._bend_params:
            apply_bend = rng.random() < bend_prob
            this_amp = bend_amplitude if apply_bend else 0.0
            min_length = np.min(motion_profile)
            dynamic_straight_fraction = min(min_length / total_length, 1.0) if total_length > 0 else 1.0
            cfg._bend_params[inst_id] = (this_amp, dynamic_straight_fraction)
        else:
            this_amp, dynamic_straight_fraction = cfg._bend_params[inst_id]
            apply_bend = this_amp > 0.0

        start_pt_j = start_pt + jitter
        end_pt_j = end_pt + jitter

        vec = end_pt_j - start_pt_j
        length = np.linalg.norm(vec)
        if length == 0:
            continue
        direction = vec / length
        perp = np.array([-direction[1], direction[0]])

        step = 0.5
        n_steps = int(length / step) + 1

        for i in range(n_steps + 1):
            t = i / n_steps
            core_pos = start_pt_j + vec * t

            if t < bend_straight_fraction or not apply_bend:
                pos = core_pos
            else:
                bend_offset = this_amp * np.sin(
                    np.pi * (t - bend_straight_fraction) / (1 - bend_straight_fraction)) * perp
                pos = core_pos + bend_offset

            local_sigma_x = sigma_x * (1 + np.random.normal(0, width_var_std))
            local_sigma_y = sigma_y * (1 + np.random.normal(0, width_var_std))

            if 0 <= pos[0] < cfg.img_size[1] and 0 <= pos[1] < cfg.img_size[0]:
                ix = min(cfg.img_size[1] - 1, max(0, int(round(pos[0]))))
                iy = min(cfg.img_size[0] - 1, max(0, int(round(pos[1]))))
                draw_tubulus(frame, pos, local_sigma_x, local_sigma_y, cfg.tubulus_contrast)
                if return_mask:
                    mask[iy, ix] = inst_id

        gt_frame.append(
            {
                "frame_idx": frame_idx,
                "start": start_pt_j.tolist(),
                "end": end_pt_j.tolist(),
                "slope": slope,
                "intercept": intercept,
                "length": float(np.linalg.norm(end_pt_j - start_pt_j)),
                "instance_id": inst_id,
            }
        )

    frame = add_fixed_spots(frame, cfg)
    frame = add_moving_spots(frame, cfg)

    frame *= decay
    frame *= vignette

    if noise_std > 0.0:
        frame += np.random.normal(0, noise_std, frame.shape).astype(np.float32)

    if invert_contrast:
        frame = 2 * bg_level - frame

    frame = apply_global_blur(frame, cfg)
    frame = np.clip(frame, 0.0, 1.0)

    frame = annotate_frame(
        frame, frame_idx, fps=cfg.fps,
        show_time=cfg.show_time,
        show_scale=cfg.show_scale,
        scale_um_per_pixel=cfg.um_per_pixel,
        scale_length_um=cfg.scale_bar_um
    )

    frame_uint8 = (frame * 255).astype(np.uint8)

    if return_mask:
        return frame_uint8, gt_frame, mask
    return frame_uint8, gt_frame, None



def generate_video(cfg: SyntheticDataConfig, base_output_dir: str):
    """
    Generate a synthetic series, its ground-truth JSON – and, if requested,
    an instance-segmentation mask video.

    Returns
    -------
    (video_path, gt_path, mask_video_path | None)
    """
    os.makedirs(base_output_dir, exist_ok=True)

    video_path = os.path.join(base_output_dir, f"series_{cfg.id:02d}.mp4")
    mask_video_path = (os.path.join(base_output_dir, f"series_{cfg.id:02d}_mask.mp4") if cfg.generate_mask else None)
    gt_path = os.path.join(base_output_dir, f"series_{cfg.id:02d}_gt.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, cfg.fps, cfg.img_size[::-1])
    mask_writer = (cv2.VideoWriter(mask_video_path, fourcc, cfg.fps, cfg.img_size[::-1]) if cfg.generate_mask else None)

    gt_accumulator: List[dict] = []

    for frame, gt_frame, mask in tqdm(
            generate_frames(cfg, return_mask=cfg.generate_mask),
            total=cfg.num_frames,
            desc=f"Series {cfg.id}"
    ):
        gt_accumulator.extend(gt_frame)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

        if cfg.generate_mask:
            # visualise mask in colour (instance ID → hue)
            mask_vis = mask_to_color(mask)
            mask_writer.write(mask_vis)

    writer.release()
    if cfg.generate_mask:
        mask_writer.release()

    save_ground_truth(gt_accumulator, gt_path)

    return video_path, gt_path, mask_video_path


def generate_frames(cfg: SyntheticDataConfig, *, return_mask: bool = False) \
        -> Generator[Tuple[np.ndarray, List[dict], Optional[np.ndarray]], None, None]:
    """
    Yield (frame_uint8, gt_for_frame, mask | None).

    `return_mask=False` keeps the signature identical to the old one, so
    callers that don’t care about masks remain unchanged.
    """
    seeds = build_motion_seeds(cfg)
    for frame_idx in range(cfg.num_frames):
        yield render_frame(cfg, seeds, frame_idx, return_mask=return_mask)


# Example usage
if __name__ == "__main__":
    output_dir = "../data/synthetic"
    config_path = "../config/best_synthetic_config.json"

    config = SyntheticDataConfig.load()
    config.id = 21
    config.to_json(config_path)
    video_path, gt_path, gt_video_path = generate_video(config, output_dir)

    print(f"Saved video: {video_path}")
    print(f"Saved gt JSON: {gt_path}")
    print(f"Saved gt Video: {gt_video_path}")
