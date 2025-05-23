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
    start_x = np.random.rand() * (IMG_SIZE[0] * 0.5) + IMG_SIZE[0] * 0.25
    start_y = np.random.rand() * (IMG_SIZE[1] * 0.5) + IMG_SIZE[1] * 0.25
    slope = np.random.uniform(-2, 2)
    intercept = start_y - slope * start_x
    end_x = start_x + MAX_LENGTH / np.sqrt(1 + slope ** 2)
    end_y = start_y + slope * (end_x - start_x)
    return np.array([end_x, end_y, slope, intercept, 0]), np.array([start_x, start_y])

def grow_shrink_seed(frame, old_info, original):
    grow_part = GROW_AMP * np.sin(GROW_FREQ * frame)
    shrink_part = SHRINK_AMP * np.sin(SHRINK_FREQ * frame)
    net_motion = grow_part - shrink_part
    end_x = old_info[0] + MOTION * net_motion
    end_y = old_info[1] + MOTION * net_motion
    return np.array([end_x, end_y, old_info[2], old_info[3], old_info[4]])

def generate_series(series_id, base_output_dir):
    video_output_path = os.path.join(base_output_dir, f"series_{series_id:02d}.mp4")
    gt_output_path = os.path.join(base_output_dir, f"series_{series_id:02d}_gt.json")
    os.makedirs(base_output_dir, exist_ok=True)

    line_info, original = get_seed()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_output_path, fourcc, FPS, IMG_SIZE)

    gt_data: List[dict] = []

    for frame in tqdm(range(NUM_FRAMES), desc=f"Generating series {series_id}"):
        img = np.zeros(IMG_SIZE, dtype=np.float32)
        line_info = grow_shrink_seed(frame, line_info, original)

        dx = 0.5 / np.sqrt(1 + line_info[2] ** 2)
        dy = line_info[2] * dx
        pos = original.copy()

        while (0 <= pos[0] < IMG_SIZE[1]) and (0 <= pos[1] < IMG_SIZE[0]) and \
              ((line_info[2] >= 0 and pos[1] <= line_info[1]) or (line_info[2] < 0 and pos[1] >= line_info[1])):
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
            "end": [line_info[0], line_info[1]],
            "slope": line_info[2],
            "intercept": line_info[3]
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

