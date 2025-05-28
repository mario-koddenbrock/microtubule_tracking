import os
from glob import glob

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoFeatureExtractor

from data_generation.config import TuningConfig
from data_generation.video import generate_frames
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
    video_files = glob(os.path.join(cfg.reference_series_dir, "*.avi")) + glob(
        os.path.join(cfg.reference_series_dir, "*.tif"))
    video_files = video_files[:cfg.num_compare_series]

    for video_path in video_files:
        frames = extract_frames(video_path)[:cfg.num_compare_frames]
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emb = compute_embedding(frame_rgb, model, extractor)
            embeddings.append(emb)

    if not embeddings:
        raise ValueError("No embeddings were extracted. Check the reference series directory and video files.")

    return np.stack([r.flatten() for r in embeddings])  # (N_ref, 256)


def cfg_to_embeddings(cfg, model, extractor) -> np.ndarray:
    """
    Generate every frame for *cfg* and return flattened embeddings.
    """
    vecs: list[np.ndarray] = []
    for frame_uint8, *_ in generate_frames(cfg):  # unpack only first item
        rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2RGB)
        vecs.append(compute_embedding(rgb, model, extractor).flatten())
    return np.stack(vecs)


def flatten_embeddings(ref_embeddings) -> np.ndarray:
    """Ensure reference embeddings are 2-D."""
    ref_arr = np.asarray(ref_embeddings)
    if ref_arr.ndim == 3:  # (N, H, W) → flatten spatial dims
        N, H, W = ref_arr.shape
        ref_arr = ref_arr.reshape(N, H * W)
    return ref_arr


def get_embedding_model(tuning_cfg):
    model = AutoModel.from_pretrained(tuning_cfg.model_name, cache_dir=tuning_cfg.hf_cache_dir)
    extractor = AutoFeatureExtractor.from_pretrained(tuning_cfg.model_name, cache_dir=tuning_cfg.hf_cache_dir)
    return extractor, model
