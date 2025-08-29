import logging
import os
from random import shuffle

import cv2
import numpy as np

logger = logging.getLogger(f"mt.{__name__}")


def convert_frame_to_uint8_bgr(frame):
    """Convert a frame to uint8 and BGR format for OpenCV."""
    # If float, scale to [0,255] and convert to uint8
    if np.issubdtype(frame.dtype, np.floating):
        frame_uint8 = np.clip(frame * 255, 0, 255).astype('uint8')
    elif frame.dtype != 'uint8':
        frame_uint8 = frame.astype('uint8')
    else:
        frame_uint8 = frame
    if len(frame_uint8.shape) == 2:
        frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
    else:
        frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
    return frame_uint8


def write_video(frames, output_path, fps, img_size=None):
    """Write a list of frames to an MP4 video file."""
    if not frames:
        return False
    height, width = frames[0].shape[:2] if img_size is None else img_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_fps = max(1, fps)
    writer = cv2.VideoWriter(output_path, fourcc, writer_fps, (width, height))
    for frame in frames:
        frame_bgr = convert_frame_to_uint8_bgr(frame)
        # Do NOT resize frames, just check shape and raise error if mismatch
        if frame_bgr.shape[0] != height or frame_bgr.shape[1] != width:
            raise ValueError(f"Frame shape {frame_bgr.shape[:2]} does not match expected size {(height, width)}. Crops must be consistent.")
        writer.write(frame_bgr)
    writer.release()
    return True


def write_png_frames(frames, output_dir, num_frames):
    """Shuffle and write a subset of frames as PNG images."""
    os.makedirs(output_dir, exist_ok=True)
    frames_shuffled = frames.copy()
    shuffle(frames_shuffled)
    for i, frame in enumerate(frames_shuffled[:num_frames]):
        frame_bgr = convert_frame_to_uint8_bgr(frame)
        frame_filename = f"{i:05d}.png"
        frame_output_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_output_path, frame_bgr)
