import os
from glob import glob
from typing import Generator, List, Tuple, Optional

import cv2
import numpy as np
import torch

from data_generation.config import SyntheticDataConfig, TuningConfig
from data_generation.sawtooth_profile import create_sawtooth_profile
from file_io.utils import extract_frames


# ---------------------------------------------------------------------------
# Frame-level renderer
# ---------------------------------------------------------------------------
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


def build_motion_seeds(cfg: SyntheticDataConfig):
    """Pre‑compute slope/intercept pairs *and* their motion profiles.

    Keeping the RNG separate from the rendering loop makes the whole pipeline
    deterministic and lets us reproduce exact sequences from a single call.
    """
    return [
        (
            get_seed(cfg.img_size, cfg.margin),
            create_sawtooth_profile(
                cfg.num_frames,
                np.random.uniform(cfg.min_length + 5, cfg.max_length),
                np.random.uniform(cfg.min_length, cfg.min_length + 10),
                noise_std=0.5,
                offset=np.random.randint(0, cfg.num_frames),
            ),
        )
        for _ in range(cfg.num_tubulus)
    ]


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
    for frame_uint8, *_ in generate_frames(cfg):     # unpack only first item
        rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2RGB)
        vecs.append(compute_embedding(rgb, model, extractor).flatten())
    return np.stack(vecs)


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


def add_gaussian(image, pos, sigma_x, sigma_y):
    if sigma_x > 0 and sigma_y > 0:
        x = np.arange(0, image.shape[1], 1)
        y = np.arange(0, image.shape[0], 1)
        x, y = np.meshgrid(x, y)
        gaussian = np.exp(-(((x - pos[0]) ** 2) / (2 * sigma_x ** 2) +
                            ((y - pos[1]) ** 2) / (2 * sigma_y ** 2)))
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


def get_seed(img_size: tuple[int, int], margin: int):
    usable_width = img_size[1] - 2 * margin
    usable_height = img_size[0] - 2 * margin
    start_x = np.random.uniform(margin, margin + usable_width)
    start_y = np.random.uniform(margin, margin + usable_height)
    slope = np.random.uniform(-1.5, 1.5)
    intercept = start_y - slope * start_x

    return np.array([slope, intercept]), np.array([start_x, start_y])


def grow_shrink_seed(frame, original, slope, motion_profile, img_size: tuple[int, int], margin: int):
    net_motion = motion_profile[frame]

    dx = net_motion / np.sqrt(1 + slope ** 2)
    dy = slope * dx

    end_x = original[0] + dx
    end_y = original[1] + dy

    # Clip to safe margin
    end_x = np.clip(end_x, margin, img_size[1] - margin)
    end_y = np.clip(end_y, margin, img_size[0] - margin)

    return np.array([end_x, end_y])


def flatten_embeddings(ref_embeddings) -> np.ndarray:
    """Ensure reference embeddings are 2-D."""
    ref_arr = np.asarray(ref_embeddings)
    if ref_arr.ndim == 3:  # (N, H, W) → flatten spatial dims
        N, H, W = ref_arr.shape
        ref_arr = ref_arr.reshape(N, H * W)
    return ref_arr
