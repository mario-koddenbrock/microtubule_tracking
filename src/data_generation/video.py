import os
from typing import List, Tuple, Optional

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from config.synthetic_data import SyntheticDataConfig
from data_generation import utils
from data_generation.spots import SpotGenerator
from data_generation.tubuli import Microtubule
from data_generation.utils import apply_random_spots
from data_generation.utils import build_motion_seeds
from file_io.utils import save_ground_truth
from skimage.color import label2rgb


def render_frame(
        cfg: SyntheticDataConfig,
        mts: list[Microtubule],
        frame_idx: int,
        fixed_spot_generator: SpotGenerator,
        moving_spot_generator: SpotGenerator,
        return_mask: bool = False,
) -> Tuple[np.ndarray, List[dict], Optional[np.ndarray]]:

    # 1) Prepare background and optional mask
    frame = np.full(cfg.img_size, cfg.background_level, dtype=np.float32)
    mask = np.zeros(cfg.img_size, dtype=np.uint16) if return_mask else None

    vignette = utils.compute_vignette(cfg)
    decay = np.exp(-frame_idx / cfg.bleach_tau) if np.isfinite(cfg.bleach_tau) else 1.0
    jitter = np.random.normal(0, cfg.jitter_px, 2) if cfg.jitter_px > 0 else np.zeros(2)
    cfg._frame_idx = frame_idx

    gt_data = []

    # 2) For each microtubule, step its length and draw
    for mt in mts:
        # A) Step to match the length profile:
        mt.step_to_length(frame_idx)

        # B) Temporarily add “jitter” to the entire chain’s base point:
        mt.base_point += jitter

        # C) Draw its wagons and collect ground truth:
        gt_info = mt.draw(frame, mask, cfg)
        gt_data.extend(gt_info)

        # D) Remove the jitter offset so it doesn’t accumulate next frame:
        mt.base_point -= jitter

    # Add background spots and noise after microtubules are drawn
    # 1. Apply fixed spots
    frame = fixed_spot_generator.apply(frame)

    # 2. Apply moving spots (at their current positions)
    frame = moving_spot_generator.apply(frame)

    # 3. Apply random spots (which are new every frame)
    frame = apply_random_spots(frame, cfg.random_spots)

    # 4. Update the state of the moving spots for the *next* frame
    moving_spot_generator.update()

    frame *= decay
    frame *= vignette
    if cfg.gaussian_noise > 0.0:
        frame += np.random.normal(0, cfg.gaussian_noise, frame.shape).astype(np.float32)


    frame = utils.apply_global_blur(frame, cfg)

    frame = utils.annotate_frame(frame, frame_idx, fps=cfg.fps, show_time=cfg.show_time, show_scale=cfg.show_scale,
                                 scale_um_per_pixel=cfg.um_per_pixel, scale_length_um=cfg.scale_bar_um)

    if cfg.invert_contrast:
        frame = 2 * cfg.background_level - frame

    frame = np.clip(frame, 0.0, 1.0)
    frame_uint8 = (frame * 255).astype(np.uint8)

    if mask.min() > 0:
        print(f"Warning: Mask has no instance IDs (min={mask.min()})")

    return (frame_uint8, gt_data, mask) if return_mask else (frame_uint8, gt_data, None)


def generate_frames(cfg: SyntheticDataConfig, *, return_mask: bool = False):
    # 1) Build a list of Microtubule objects instead of raw “seeds”:
    mts = []

    for idx, (start_pt, motion_profile) in enumerate(build_motion_seeds(cfg), start=1):
        # 1) Randomize base orientation as before:
        base_orient = np.random.uniform(0.0, 2 * np.pi)

        # 2) Draw a per‐microtubule angle_change_prob ∈ [0, cfg.max_angle_change_prob]:
        angle_change_prob = np.random.uniform(0.0, cfg.max_angle_change_prob)

        # 3) Draw base-wagon length:
        base_len = np.random.uniform(
            cfg.min_base_wagon_length,
            cfg.max_base_wagon_length
        )

        # 4) Draw per-wagon length bounds:
        min_wagon_length = np.random.uniform(
            cfg.min_wagon_length_min,
            cfg.min_wagon_length_max
        )
        max_wagon_length = np.random.uniform(
            cfg.max_wagon_length_min,
            cfg.max_wagon_length_max
        )

        # 5) Instantiate Microtubule with its own angle_change_prob:
        mt = Microtubule(
            base_point=start_pt,
            base_orientation=base_orient,
            base_wagon_length=base_len,
            profile=motion_profile,
            max_num_wagons=cfg.max_num_wagons,
            max_angle=cfg.max_angle,
            angle_change_prob=angle_change_prob,
            min_wagon_length=min_wagon_length,
            max_wagon_length=max_wagon_length,
            instance_id=idx,
        )
        mt.instance_id = idx
        mts.append(mt)

    fixed_spot_generator = SpotGenerator(cfg.fixed_spots, cfg.img_size)
    moving_spot_generator = SpotGenerator(cfg.moving_spots, cfg.img_size)

    # 2) For each frame, step each microtubule and draw it:
    for frame_idx in range(cfg.num_frames):
        frame, gt_data, mask = render_frame(cfg, mts, frame_idx, fixed_spot_generator, moving_spot_generator,
                                           return_mask=return_mask)
        yield frame, gt_data, mask


# In video.py

def generate_video(cfg: SyntheticDataConfig, base_output_dir: str):
    """
    Generates a synthetic video sequence and saves all outputs in a streaming fashion
    to minimize memory usage.

    Saves the following files based on a consistent naming scheme:
    - Primary Data:
        - series_{id}_video.tif (Grayscale uint8 video)
        - series_{id}_masks.tif (Integer uint16 instance masks)
    - Previews:
        - series_{id}_video_preview.mp4
        - series_{id}_video_preview.gif
        - series_{id}_masks_preview.mp4
    - Metadata:
        - series_{id}_gt.json
    """
    os.makedirs(base_output_dir, exist_ok=True)

    # --- 1. Define all output paths using the new naming scheme ---
    base_name = f"series_{cfg.id}"
    video_tiff_path = os.path.join(base_output_dir, f"{base_name}_video.tif")
    masks_tiff_path = os.path.join(base_output_dir, f"{base_name}_masks.tif")
    video_mp4_path = os.path.join(base_output_dir, f"{base_name}_video_preview.mp4")
    masks_mp4_path = os.path.join(base_output_dir, f"{base_name}_masks_preview.mp4")
    gif_path = os.path.join(base_output_dir, f"{base_name}_video_preview.gif")
    gt_json_path = os.path.join(base_output_dir, f"{base_name}_gt.json")

    # --- 2. Initialize writers within a try...finally block ---
    # This ensures all files are properly closed even if an error occurs.

    # imageio writers for high-quality data (TIFF) and GIF
    video_tiff_writer = imageio.get_writer(video_tiff_path, format='TIFF')
    gif_writer = imageio.get_writer(gif_path, fps=cfg.fps)
    mask_tiff_writer = imageio.get_writer(masks_tiff_path, format='TIFF') if cfg.generate_mask else None

    # cv2 writers for compressed MP4 previews
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    img_h, img_w = cfg.img_size
    video_mp4_writer = cv2.VideoWriter(video_mp4_path, fourcc, cfg.fps, (img_w, img_h))
    mask_mp4_writer = cv2.VideoWriter(masks_mp4_path, fourcc, cfg.fps, (img_w, img_h)) if cfg.generate_mask else None

    try:
        all_gt_data = []
        cfg._bend_params = {}  # Reset per-video bending memory

        print(f"Generating and writing {cfg.num_frames} frames for Series {cfg.id}...")

        # --- 3. Process and write each frame one-by-one ---
        for frame_img, gt_data_for_frame, mask_img in tqdm(
                generate_frames(cfg, return_mask=cfg.generate_mask),
                total=cfg.num_frames
        ):
            # A. Accumulate JSON data (this is small, so it's fine to keep in memory)
            all_gt_data.extend(gt_data_for_frame)

            # B. Write the main video frames
            video_tiff_writer.append_data(frame_img)  # Raw uint8 data

            # For previews, convert grayscale to color
            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)
            video_mp4_writer.write(frame_bgr)

            frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2RGB)
            gif_writer.append_data(frame_rgb)

            # C. Write the mask frames (if enabled)
            if cfg.generate_mask and mask_img is not None:
                mask_tiff_writer.append_data(mask_img)  # Raw uint16 data

                # Create and write the colorized preview for the mask
                mask_vis_float = label2rgb(mask_img, bg_label=0)
                mask_vis_uint8 = (mask_vis_float * 255).astype(np.uint8)
                mask_vis_bgr = cv2.cvtColor(mask_vis_uint8, cv2.COLOR_RGB2BGR)
                mask_mp4_writer.write(mask_vis_bgr)

        # --- 4. Save the collected ground truth data after the loop ---
        save_ground_truth(all_gt_data, gt_json_path)

    finally:
        # --- 5. Close all writers to finalize the files ---
        print("Closing all file writers...")
        video_tiff_writer.close()
        gif_writer.close()
        video_mp4_writer.release()
        if mask_tiff_writer:
            mask_tiff_writer.close()
        if mask_mp4_writer:
            mask_mp4_writer.release()
        print("All files saved successfully.")

    return video_tiff_path, gt_json_path, masks_tiff_path


