import os
from typing import Generator, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

from data_generation.config import SyntheticDataConfig
from data_generation.utils import grow_shrink_seed, add_gaussian, normalize_image, poisson_noise, build_motion_seeds
from file_io.utils import save_ground_truth
from plotting.plotting import mask_to_color


def render_frame(
        cfg: SyntheticDataConfig,
        seeds,
        frame_idx: int,
        *,
        return_mask: bool = False,
) -> Tuple[np.ndarray, List[dict], Optional[np.ndarray]]:
    """Render **one** frame and (optionally) its instance-segmentation mask.

    Parameters
    ----------
    cfg : SyntheticDataConfig
    seeds : list
        Output of `build_motion_seeds()` –  [(slope/intercept, start_pt), motion_profile] …
    frame_idx : int
    return_mask : bool, optional
        If *True* an additional array of shape ``cfg.img_size`` is returned in
        which each pixel holds the **instance ID** (0 = background, 1-based for
        every microtubule).

    Returns
    -------
    frame_uint8 : np.ndarray
        Grayscale image in 0-255 range, dtype ``uint8``.
    gt_frame : list[dict]
        One record per tubule for this frame.
    mask_uint16 | None
        Only when *return_mask* is *True*; same spatial size as the frame, each
        pixel labelled with the instance ID.
    """

    img = np.zeros(cfg.img_size, dtype=np.float32)
    mask: Optional[np.ndarray] = None
    if return_mask:
        mask = np.zeros(cfg.img_size, dtype=np.uint16)  # background = 0

    gt_frame: List[dict] = []

    # Iterate over each synthetic tubule (instance) ------------------------
    for inst_id, ((slope, intercept), start_pt), motion_profile in (
            (idx + 1, *seed) for idx, seed in enumerate(seeds)
    ):
        # --- motion -------------------------------------------------------
        end_pt = grow_shrink_seed(
            frame_idx, start_pt, slope, motion_profile, cfg.img_size, cfg.margin
        )

        # --- draw line of Gaussians --------------------------------------
        dx = 0.5 / np.hypot(1, slope)
        dy = slope * dx
        pos = start_pt.copy()
        while (
                (dx > 0 and pos[0] <= end_pt[0]) or (dx < 0 and pos[0] >= end_pt[0])
        ) and (
                (dy > 0 and pos[1] <= end_pt[1]) or (dy < 0 and pos[1] >= end_pt[1])
        ) and (0 <= pos[0] < cfg.img_size[1]) and (0 <= pos[1] < cfg.img_size[0]):
            add_gaussian(img, pos, cfg.sigma_x, cfg.sigma_y)
            if return_mask:
                mask[int(round(pos[1])), int(round(pos[0]))] = inst_id
            pos[0] += dx
            pos[1] += dy

        # --- ground-truth record -----------------------------------------
        gt_frame.append(
            {
                "frame_idx": frame_idx,
                "start": start_pt.tolist(),
                "end": end_pt.tolist(),
                "slope": slope,
                "intercept": intercept,
                "length": float(np.linalg.norm(end_pt - start_pt)),
                "instance_id": inst_id,
            }
        )

    # ---------------------------------------------------------------------
    img = normalize_image(img)
    noisy_img = poisson_noise(img, cfg.snr)
    frame_uint8 = (noisy_img * 255).astype(np.uint8)

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
    config_path = "../config/synthetic_config.json"

    config = SyntheticDataConfig.load(config_path)
    video_path, gt_path, gt_video_path = generate_video(config, output_dir)

    print(f"Saved video: {video_path}")
    print(f"Saved gt JSON: {gt_path}")
    print(f"Saved gt Video: {gt_video_path}")
