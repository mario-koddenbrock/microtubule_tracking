import os
from typing import List

from tqdm import tqdm

from data_generation.config import SyntheticDataConfig
from data_generation.utils import generate_frames
from file_io.utils import save_ground_truth
from file_io.video import write_video


def run_series(cfg: SyntheticDataConfig, base_output_dir: str):
    """Glue together *pure* generators with *impure* writers."""

    os.makedirs(base_output_dir, exist_ok=True)
    video_path = os.path.join(base_output_dir, f"series_{cfg.id:02d}.mp4")
    gt_path = os.path.join(base_output_dir, f"series_{cfg.id:02d}_gt.json")

    gt_accumulator: List[dict] = []

    def frame_only_gen():
        for frame, gt_frame in tqdm(
                generate_frames(cfg), total=cfg.num_frames, desc=f"Series {cfg.id}"
        ):
            gt_accumulator.extend(gt_frame)
            yield frame

    write_video(frame_only_gen(), video_path, cfg.fps, cfg.img_size)
    save_ground_truth(gt_accumulator, gt_path)

    return video_path, gt_path


# Example usage
if __name__ == "__main__":
    output_dir = "../data/synthetic"
    config_path = "../config/synthetic_config.json"

    config = SyntheticDataConfig.load(config_path)
    video_path, gt_path = run_series(config, output_dir)

    print(f"Saved video: {video_path}")
    print(f"Saved gt: {gt_path}")
