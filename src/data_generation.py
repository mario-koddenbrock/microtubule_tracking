import numpy as np
import os
import cv2
import json
from typing import List
from tqdm import tqdm

# --- Parameters ---
IMG_SIZE = (256, 256)
FPS = 10
NUM_FRAMES = 60 * FPS  # 1 minute duration
SNR = 5
GROW_AMP = 2.0
GROW_FREQ = 0.05
SHRINK_AMP = 4.0
SHRINK_FREQ = 0.25
MOTION = 2.1
MAX_LENGTH = 50
SIGMA = [2, 2]
NUM_SERIES = 3  # number of synthetic videos to generate
MARGIN = 5  # margin from borders


def create_sawtooth_profile(num_frames, max_length, noise_std=1.0):
    profile = []
    grow_len = max_length
    shrink_len = max_length * 0.9
    grow_frames = int(num_frames / 6)
    shrink_frames = int(grow_frames / 3)
    t = 0
    while len(profile) < num_frames:
        # Grow phase (slow)
        for i in range(grow_frames):
            if len(profile) >= num_frames: break
            val = i / grow_frames * grow_len + np.random.normal(0, noise_std)
            profile.append(val)
        # Shrink phase (fast)
        for i in range(shrink_frames):
            if len(profile) >= num_frames: break
            val = grow_len - i / shrink_frames * shrink_len + np.random.normal(0, noise_std)
            profile.append(val)
    return profile[:num_frames]


def add_gaussian(image, pos, sigma):
    x = np.arange(0, image.shape[1], 1)
    y = np.arange(0, image.shape[0], 1)
    x, y = np.meshgrid(x, y)
    gaussian = np.exp(-(((x - pos[0]) ** 2) / (2 * sigma[0] ** 2) +
                        ((y - pos[1]) ** 2) / (2 * sigma[1] ** 2)))
    image += gaussian
    return image


def normalize_image(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min + 1e-8)


def poisson_noise(image, snr):
    max_val = np.max(image)
    noisy = np.random.poisson(image * snr) / snr
    return np.clip(noisy / max_val if max_val > 0 else image, 0, 1)


def get_seed():
    usable_width = IMG_SIZE[1] - 2 * MARGIN
    usable_height = IMG_SIZE[0] - 2 * MARGIN
    start_x = np.random.uniform(MARGIN, MARGIN + usable_width)
    start_y = np.random.uniform(MARGIN, MARGIN + usable_height)
    slope = np.random.uniform(-1.5, 1.5)
    intercept = start_y - slope * start_x

    return np.array([slope, intercept]), np.array([start_x, start_y])


def grow_shrink_seed(frame, original, slope, motion_profile):
    net_motion = motion_profile[frame]

    dx = net_motion / np.sqrt(1 + slope ** 2)
    dy = slope * dx

    end_x = original[0] + dx
    end_y = original[1] + dy

    # Clip to safe margin
    end_x = np.clip(end_x, MARGIN, IMG_SIZE[1] - MARGIN)
    end_y = np.clip(end_y, MARGIN, IMG_SIZE[0] - MARGIN)

    return np.array([end_x, end_y])


def generate_series(series_id, base_output_dir, num_microtubules=3):
    video_output_path = os.path.join(base_output_dir, f"series_{series_id:02d}.mp4")
    gt_output_path = os.path.join(base_output_dir, f"series_{series_id:02d}_gt.json")
    os.makedirs(base_output_dir, exist_ok=True)

    seeds = [(get_seed(), create_sawtooth_profile(NUM_FRAMES, MAX_LENGTH, noise_std=1.5)) for _ in
             range(num_microtubules)]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_output_path, fourcc, FPS, IMG_SIZE[::-1])

    gt_data: List[dict] = []

    for frame in tqdm(range(NUM_FRAMES), desc=f"Generating series {series_id}"):
        img = np.zeros(IMG_SIZE, dtype=np.float32)

        for (slope_intercept, original), motion_profile in seeds:
            slope, intercept = slope_intercept
            end = grow_shrink_seed(frame, original, slope, motion_profile)

            dx = 0.5 / np.sqrt(1 + slope ** 2)
            dy = slope * dx
            pos = original.copy()
            while (
                    (dx > 0 and pos[0] <= end[0]) or (dx < 0 and pos[0] >= end[0])
            ) and (
                    (dy > 0 and pos[1] <= end[1]) or (dy < 0 and pos[1] >= end[1])
            ) and (0 <= pos[0] < IMG_SIZE[1]) and (0 <= pos[1] < IMG_SIZE[0]):
                add_gaussian(img, pos, SIGMA)
                pos[0] += dx
                pos[1] += dy

        img = normalize_image(img)
        noisy_img = poisson_noise(img, SNR)
        frame_uint8 = (noisy_img * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR))

        gt_data.append({
            "frame": frame,
            "start": original.tolist(),
            "end": end.tolist(),
            "slope": slope,
            "intercept": intercept
        })

    writer.release()
    with open(gt_output_path, 'w') as f:
        json.dump(gt_data, f, indent=2)

    return video_output_path, gt_output_path


# Example usage
if __name__ == "__main__":
    output_dir = "../data/synthetic"
    for idx in range(NUM_SERIES):
        generate_series(idx, output_dir)
