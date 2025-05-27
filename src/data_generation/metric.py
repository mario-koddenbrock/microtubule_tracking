import os
from glob import glob

import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from data_generation.config import TuningConfig, SyntheticDataConfig
from data_generation.main import generate_series
from file_io.utils import extract_frames


def compute_embedding(image, model, feature_extractor):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "last_hidden_state"):
        embedding = outputs.last_hidden_state.mean(dim=1)
    elif hasattr(outputs, "pooler_output"):
        embedding = outputs.pooler_output
    else:
        raise ValueError("Model output does not contain a usable embedding.")

    return embedding.squeeze().cpu().numpy()


def load_reference_embeddings(cfg: TuningConfig, model, extractor):
    embeddings = []
    video_files = glob(os.path.join(cfg.reference_series_dir, "*.avi")) + glob(os.path.join(cfg.reference_series_dir, "*.tif"))
    video_files = video_files[:cfg.num_compare_series]

    for video_path in video_files:
        frames = extract_frames(video_path)[:cfg.num_compare_frames]
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emb = compute_embedding(frame_rgb, model, extractor)
            embeddings.append(emb)

    if not embeddings:
        raise ValueError("No embeddings were extracted. Check the reference series directory and video files.")

    return embeddings

def evaluate(cfg_dict, ref_embeddings, cfg: TuningConfig, model, extractor):
    synth_cfg = SyntheticDataConfig(**cfg_dict)
    synth_cfg.id = 0
    output_dir = "temp_eval_output"
    generate_series(synth_cfg, output_dir)

    video_path = os.path.join(output_dir, "series_00.mp4")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return -1.0

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    emb = compute_embedding(frame_rgb, model, extractor)
    scores = [cosine_similarity([emb], [r])[0, 0] for r in ref_embeddings]
    return float(np.mean(scores))