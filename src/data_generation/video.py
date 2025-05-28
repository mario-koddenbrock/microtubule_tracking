import os
from typing import List, Tuple, Optional, Generator

import cv2
import numpy as np
from tqdm import tqdm

from data_generation.config import SyntheticDataConfig
from data_generation.utils import (
    add_gaussian,
    grow_shrink_seed,
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
    """Render **one** frame

    Extra photophysics simulated here:
        • background pedestal (& optional vignetting)
        • additive Gaussian sensor noise
        • photobleaching (single exponential)
        • stage jitter (global xy shift, per frame)
        • instance‑segmentation mask (optional)
    """

    # 0️⃣  Convenience access to new cfg fields -----------------------------
    bg_level = float(getattr(cfg, "background_level", 0.0))  # 0‑1 scale
    noise_std = float(getattr(cfg, "gaussian_noise", 0.0))  # 0‑1 scale
    bleach_tau = float(getattr(cfg, "bleach_tau", np.inf))  # frames
    jitter_px = float(getattr(cfg, "jitter_px", 0.0))
    vignette_k = float(getattr(cfg, "vignetting_strength", 0.0))

    # contrast control: set cfg.invert_contrast = True for dark tubules
    invert_contrast = bool(getattr(cfg, "invert_contrast", False))

    # σ parameters ----------------------------------------------------------
    sigma_x = getattr(cfg, "sigma_x", None)
    sigma_y = getattr(cfg, "sigma_y", None)


    # 1️⃣  Base image initialisation ---------------------------------------
    img = np.full(cfg.img_size, bg_level, dtype=np.float32)
    mask: Optional[np.ndarray] = None
    if return_mask:
        mask = np.zeros(cfg.img_size, dtype=np.uint16)  # background = 0

    # Pre‑compute vignetting mask (per series or lazily per frame) ---------
    if vignette_k > 0.0:
        yy, xx = np.mgrid[:cfg.img_size[0], :cfg.img_size[1]]
        norm_x = (xx - cfg.img_size[1] / 2) / (cfg.img_size[1] / 2)
        norm_y = (yy - cfg.img_size[0] / 2) / (cfg.img_size[0] / 2)
        vignette = 1.0 - vignette_k * (norm_x ** 2 + norm_y ** 2)
        vignette = np.clip(vignette, 0.5, 1.0)
    else:
        vignette = 1.0  # scalar broadcast

    # Photobleaching decay factor ------------------------------------------
    decay = np.exp(-frame_idx / bleach_tau) if np.isfinite(bleach_tau) else 1.0

    # Stage jitter ----------------------------------------------------------
    jitter = np.random.normal(0, jitter_px, 2) if jitter_px > 0 else np.zeros(2)

    gt_frame: List[dict] = []

    # 2️⃣  Draw every tubule instance --------------------------------------
    for inst_id, ((slope, intercept), start_pt), motion_profile in (
            (idx + 1, *seed) for idx, seed in enumerate(seeds)
    ):
        end_pt = grow_shrink_seed(
            frame_idx, start_pt, slope, motion_profile, cfg.img_size, cfg.margin
        )

        # Apply global jitter to both endpoints
        start_pt_j = start_pt + jitter
        end_pt_j = end_pt + jitter

        dx = 0.5 / np.hypot(1, slope)
        dy = slope * dx
        pos = start_pt_j.copy()

        while (
                (dx > 0 and pos[0] <= end_pt_j[0]) or (dx < 0 and pos[0] >= end_pt_j[0])
        ) and (
                (dy > 0 and pos[1] <= end_pt_j[1]) or (dy < 0 and pos[1] >= end_pt_j[1])
        ) and (0 <= pos[0] < cfg.img_size[1]) and (0 <= pos[1] < cfg.img_size[0]):
            add_gaussian(img, pos, sigma_x, sigma_y)
            if return_mask:
                mask[int(round(pos[1])), int(round(pos[0]))] = inst_id
            pos[0] += dx
            pos[1] += dy

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

    # 3️⃣  Photophysics post‑processing ------------------------------------
    img *= decay
    img *= vignette

    if noise_std > 0.0:
        img += np.random.normal(0, noise_std, img.shape).astype(np.float32)

    # ── invert contrast so tubules become *dark* on bright bg ────────────
    if invert_contrast:
        img = 2 * bg_level - img   # symmetric around the pedestal

    # final clamp & quantise
    img = np.clip(img, 0.0, 1.0)

    if noise_std > 0.0:
        img += np.random.normal(0, noise_std, img.shape).astype(np.float32)

    img = np.clip(img, 0.0, 1.0)
    frame_uint8 = (img * 255).astype(np.uint8)

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
    config.to_json(config_path)
    video_path, gt_path, gt_video_path = generate_video(config, output_dir)

    print(f"Saved video: {video_path}")
    print(f"Saved gt JSON: {gt_path}")
    print(f"Saved gt Video: {gt_video_path}")
