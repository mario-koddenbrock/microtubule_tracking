import os
from typing import List

import cv2
from tqdm import tqdm

from data_generation.config import SyntheticDataConfig
from data_generation.utils import generate_frames
from file_io.utils import save_ground_truth
from plotting.plotting import mask_to_color


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


# Example usage
if __name__ == "__main__":
    output_dir = "../data/synthetic"
    config_path = "../config/synthetic_config.json"

    config = SyntheticDataConfig.load(config_path)
    video_path, gt_path, gt_video_path = run_series(config, output_dir)

    print(f"Saved video: {video_path}")
    print(f"Saved gt JSON: {gt_path}")
    print(f"Saved gt Video: {gt_video_path}")
