import os
from glob import glob

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoFeatureExtractor, CLIPModel, CLIPImageProcessor

from data_generation.config import TuningConfig
from data_generation.video import generate_frames
from file_io.utils import extract_frames


def compute(image, model, processor):
    """
    image : RGB np.ndarray or PIL.Image
    processor : AutoFeatureExtractor | CLIPImageProcessor
    """
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        if isinstance(model, CLIPModel):
            # CLIP gives you a 512-dim vector per image
            embedding = model.get_image_features(**inputs)            # (1, 512)
        else:
            outputs = model(**inputs)
            if hasattr(outputs, "last_hidden_state"):
                embedding = outputs.last_hidden_state.mean(dim=1)     # (1, D)
            elif hasattr(outputs, "pooler_output"):
                embedding = outputs.pooler_output                     # (1, D)
            else:
                raise ValueError("Model output does not contain a usable embedding.")

    return embedding.squeeze().cpu().numpy()


def load_references(cfg: TuningConfig, model, extractor):
    embeddings = []
    video_files = glob(os.path.join(cfg.reference_series_dir, "*.avi")) + glob(
        os.path.join(cfg.reference_series_dir, "*.tif"))
    video_files = video_files[:cfg.num_compare_series]

    for video_idx, video_path in enumerate(video_files):
        frames = extract_frames(video_path)[:cfg.num_compare_frames]

        print(f"Reference {video_idx + 1}/{len(video_files)}: {os.path.basename(video_path)} - {len(frames)} frames - {frames[0].shape if frames else 'No frames'}")
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emb = compute(frame_rgb, model, extractor)
            embeddings.append(emb)

    if not embeddings:
        raise ValueError("No embeddings were extracted. Check the reference series directory and video files.")

    return np.stack([r.flatten() for r in embeddings])  # (N_ref, 256)


def from_cfg(cfg, model, extractor) -> np.ndarray:
    """
    Generate every frame for *cfg* and return flattened embeddings.
    """
    vecs: list[np.ndarray] = []
    for frame_uint8, *_ in generate_frames(cfg):  # unpack only first item
        rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2RGB)
        vecs.append(compute(rgb, model, extractor).flatten())
    return np.stack(vecs)


def flatten(ref_embeddings) -> np.ndarray:
    """Ensure reference embeddings are 2-D."""
    ref_arr = np.asarray(ref_embeddings)
    if ref_arr.ndim == 3:  # (N, H, W) → flatten spatial dims
        N, H, W = ref_arr.shape
        ref_arr = ref_arr.reshape(N, H * W)
    return ref_arr


def get_model(tuning_cfg):
    if "clip" in tuning_cfg.model_name.lower():
        model = CLIPModel.from_pretrained(tuning_cfg.model_name, cache_dir=tuning_cfg.hf_cache_dir)
        extractor = CLIPImageProcessor.from_pretrained(tuning_cfg.model_name, cache_dir=tuning_cfg.hf_cache_dir)
    else:
        model = AutoModel.from_pretrained(tuning_cfg.model_name, cache_dir=tuning_cfg.hf_cache_dir)
        extractor = AutoFeatureExtractor.from_pretrained(tuning_cfg.model_name, cache_dir=tuning_cfg.hf_cache_dir)

    model.eval()  # no gradients
    return extractor, model
