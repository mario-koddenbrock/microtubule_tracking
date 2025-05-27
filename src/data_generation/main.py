import numpy as np
import os
import cv2
import json
from typing import List
from tqdm import tqdm

from data_generation.sawtooth_profile import create_sawtooth_profile
from data_generation.utils import get_seed, grow_shrink_seed, add_gaussian, normalize_image, \
    poisson_noise
from data_generation.config import SyntheticDataConfig


def generate_series(config:SyntheticDataConfig, base_output_dir:str):

    video_output_path = os.path.join(base_output_dir, f"series_{config.id:02d}.mp4")
    gt_output_path = os.path.join(base_output_dir, f"series_{config.id:02d}_gt.json")
    os.makedirs(base_output_dir, exist_ok=True)

    seeds = [
        (
            get_seed(config.img_size, config.margin),
            create_sawtooth_profile(
                config.num_frames,
                np.random.uniform(config.min_length + 5, config.max_length),
                np.random.uniform(config.min_length, config.min_length + 10),
                noise_std=0.5,
                offset=np.random.randint(0, config.num_frames)
            )
        ) for _ in range(config.num_tubulus)
    ]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_output_path, fourcc, config.fps, config.img_size[::-1]) # todo check rgb vs. grayscale

    gt_data: List[dict] = []

    for frame_idx in tqdm(range(config.num_frames), desc=f"Generating series {config.id}"):
        img = np.zeros(config.img_size, dtype=np.float32)

        for (slope_intercept, start_pt), motion_profile in seeds:
            slope, intercept = slope_intercept
            end_pt = grow_shrink_seed(frame_idx, start_pt, slope, motion_profile, config.img_size, config.margin)

            dx = 0.5 / np.sqrt(1 + slope ** 2)
            dy = slope * dx
            pos = start_pt.copy()
            while (
                    (dx > 0 and pos[0] <= end_pt[0]) or (dx < 0 and pos[0] >= end_pt[0])
            ) and (
                    (dy > 0 and pos[1] <= end_pt[1]) or (dy < 0 and pos[1] >= end_pt[1])
            ) and (0 <= pos[0] < config.img_size[1]) and (0 <= pos[1] < config.img_size[0]):
                add_gaussian(img, pos, config.sigma)
                pos[0] += dx
                pos[1] += dy

            gt_data.append({
                    "frame_idx": frame_idx,
                    "start": start_pt.tolist(),
                    "end": end_pt.tolist(),
                    "slope": slope,
                    "intercept": intercept,
                    "length": float(np.linalg.norm(end_pt - start_pt))
                })

        img = normalize_image(img)
        noisy_img = poisson_noise(img, config.snr)
        frame_uint8 = (noisy_img * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR))



    writer.release()
    with open(gt_output_path, 'w') as f:
        json.dump(gt_data, f, indent=2)

    return video_output_path, gt_output_path


# Example usage
if __name__ == "__main__":
    output_dir = "../data/synthetic"
    config_path = "../config/synthetic_config.json"

    config = SyntheticDataConfig.load(config_path)

    generate_series(config, output_dir)
