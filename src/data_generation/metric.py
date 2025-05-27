import os

import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from data_generation.config import TuningConfig, SyntheticDataConfig
from data_generation.main import generate_series


def compute_embedding(image, model, feature_extractor):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def load_reference_embeddings(cfg: TuningConfig, model, extractor):
    embeddings = []
    for i in range(cfg.num_compare_series):
        video_path = os.path.join(cfg.reference_series_dir, f"series_{i:02d}.mp4")
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(cfg.num_compare_frames):
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        for frame in frames:
            emb = compute_embedding(frame, model, extractor)
            embeddings.append(emb)
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