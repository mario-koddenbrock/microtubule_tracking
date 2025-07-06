import logging
from typing import Generator, Tuple

import cv2
import numpy as np


logger = logging.getLogger(f"microtuble_tracking.{__name__}")


def write_video(frames: Generator[np.ndarray, None, None], output_path: str, fps: int, img_size: Tuple[int, int]):
    """
    Writes a sequence of frames from a generator to a video file.

    Args:
        frames (Generator[np.ndarray, None, None]): A generator yielding numpy arrays (frames).
                                                   Assumes frames are grayscale (H, W) or RGB (H, W, 3).
        output_path (str): The full path to the output video file (e.g., "output/video.mp4").
        fps (int): Frames per second for the output video.
        img_size (Tuple[int, int]): The (height, width) of the frames.
    """
    logger.info(f"Starting video writing to: {output_path}")
    logger.debug(f"Video parameters: FPS={fps}, Image Size={img_size} (H, W).")

    # OpenCV's VideoWriter expects (width, height)
    video_width, video_height = img_size[1], img_size[0]

    # Define the codec. 'mp4v' is a common choice for MP4.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    logger.debug(f"Video codec FOURCC: '{fourcc}'.")

    writer = None
    try:
        # Initialize VideoWriter
        writer = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))

        if not writer.isOpened():
            logger.error(
                f"Failed to open video writer for path: {output_path}. Check path, codecs, or file permissions.")
            raise IOError(f"Could not open video writer for {output_path}")

        frame_count = 0
        for i, frame in enumerate(frames):
            if frame.ndim == 2:  # Grayscale frame, convert to BGR for output
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                logger.debug(f"Frame {i}: Converted grayscale to BGR for writing.")
            elif frame.ndim == 3 and frame.shape[2] == 3:  # Already RGB or BGR
                # Assuming incoming frames are RGB (e.g., from generated data)
                # OpenCV writes BGR, so convert RGB to BGR if necessary
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                logger.debug(f"Frame {i}: Converted RGB to BGR for writing.")
            else:
                logger.warning(
                    f"Frame {i}: Unexpected frame dimensions ({frame.shape}). Attempting to write as is (may fail).")
                processed_frame = frame

            if processed_frame.shape[0] != video_height or processed_frame.shape[1] != video_width:
                logger.warning(
                    f"Frame {i}: Dimension mismatch (expected {video_height}x{video_width}, got {processed_frame.shape[0]}x{processed_frame.shape[1]}). Resizing.")
                processed_frame = cv2.resize(processed_frame, (video_width, video_height))

            writer.write(processed_frame)
            frame_count += 1
            if frame_count % (fps * 10) == 0:  # Log progress every 10 seconds of video
                logger.info(f"Wrote {frame_count} frames to {output_path}...")

        logger.info(f"Finished writing {frame_count} frames to: {output_path}")

    except Exception as e:
        logger.error(f"An error occurred during video writing to {output_path}: {e}", exc_info=True)
        raise  # Re-raise the exception to indicate failure

    finally:
        if writer:
            writer.release()
            logger.debug(f"VideoWriter released for {output_path}.")
        else:
            logger.warning(f"VideoWriter was not initialized for {output_path}, nothing to release.")