import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from analysis.kymographs import generate_kymographs
from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation.optimization.metrics import similarity, precompute_matric_args


logger = logging.getLogger(f"mt.{__name__}")


def visualize_embeddings(cfg: SyntheticDataConfig, tuning_cfg: TuningConfig,
                         ref_embeddings: np.ndarray, synthetic_embeddings: np.ndarray,
                         output_dir: str = "plots/"):

    logger.info(f"Starting visualization of embeddings for config ID: {cfg.id}...")
    logger.debug(
        f"Reference embeddings shape: {ref_embeddings.shape}, Synthetic embeddings shape: {synthetic_embeddings.shape}.")


    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, f"heatmap_{cfg.id}.png")

    try:
        plot_similarity_matrix(tuning_cfg, ref_embeddings, synthetic_embeddings, save_to=heatmap_path)
        logger.info(f"Similarity heatmap saved to {heatmap_path}")
    except Exception as e:
        logger.error(f"Failed to generate similarity heatmap: {e}", exc_info=True)


    precomputed_kwargs = precompute_matric_args(tuning_cfg, ref_embeddings)
    colour = np.array([similarity(
        tuning_cfg=tuning_cfg,
        ref_embeddings=ref_embeddings,
        synthetic_embeddings=synthetic_embeddings[i, :].reshape(1, -1),
        **precomputed_kwargs,
    ) for i in tqdm(range(synthetic_embeddings.shape[0]), desc="Calculating per-frame similarity")])
    logger.debug(
        f"Per-frame similarity 'colour' array shape: {colour.shape}, Min={colour.min():.4f}, Max={colour.max():.4f}.")

    colour[np.isinf(colour)] = 0

    # 3. Perform dimensionality reduction and plot 2D projection
    projection_path = os.path.join(output_dir, f"projection_{cfg.id}.png")  # Changed from tsne_ to generic projection_
    projection_method_name = "PCA"  # Default method name


    logger.info(f"Performing {projection_method_name} projection for 2D plot.")
    ref_2d, synthetic_2d = pca_projection(ref_embeddings, synthetic_embeddings)
    logger.debug(f"Projection complete. Ref 2D shape: {ref_2d.shape}, Synthetic 2D shape: {synthetic_2d.shape}.")

    plot_2d_projection(ref_2d, synthetic_2d, colour, save_to=projection_path, method_name=projection_method_name)
    logger.info(f"2D embedding projection plot saved to {projection_path}")




def tsne_projection(ref_embeddings: np.ndarray, synthetic_embeddings: np.ndarray, similarity_metric: str,
                    perplexity: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs t-SNE projection on reference and synthetic embeddings.

    Args:
        ref_embeddings (np.ndarray): Embeddings from reference images.
        synthetic_embeddings (np.ndarray): Embeddings from synthetic images.
        similarity_metric (str): Metric to use for t-SNE.
        perplexity (int): Perplexity for t-SNE.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 2D projected reference and synthetic embeddings.
    """
    logger.debug(f"Starting t-SNE projection with metric '{similarity_metric}' and perplexity {perplexity}.")
    logger.debug(
        f"Ref embeddings shape: {ref_embeddings.shape}, Synthetic embeddings shape: {synthetic_embeddings.shape}.")

    all_vecs = np.vstack([ref_embeddings, synthetic_embeddings])

    # Perplexity must be less than the number of samples
    safe_perplexity = min(perplexity, max(1, ref_embeddings.shape[0] - 1), max(1, synthetic_embeddings.shape[0] - 1))
    if safe_perplexity != perplexity:
        logger.warning(f"Adjusted t-SNE perplexity from {perplexity} to {safe_perplexity} due to insufficient samples.")

    if all_vecs.shape[0] <= 1:
        logger.warning("Not enough samples for t-SNE. Returning original embeddings as is.")
        return ref_embeddings, synthetic_embeddings  # Or handle as error/empty

    try:
        tsne = TSNE(n_components=2, metric=similarity_metric, perplexity=safe_perplexity, init="pca", random_state=42)
        all_2d = tsne.fit_transform(all_vecs)
        logger.debug("t-SNE fitting and transformation complete.")
        return np.split(all_2d, [len(ref_embeddings)], axis=0)
    except Exception as e:
        logger.error(f"Error during t-SNE projection: {e}", exc_info=True)
        # Return dummy values or re-raise based on desired error handling
        return np.zeros((ref_embeddings.shape[0], 2)), np.zeros((synthetic_embeddings.shape[0], 2))


def pca_projection(ref_embeddings: np.ndarray, synthetic_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs PCA projection on reference and synthetic embeddings.

    Args:
        ref_embeddings (np.ndarray): Embeddings from reference images.
        synthetic_embeddings (np.ndarray): Embeddings from synthetic images.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 2D projected reference and synthetic embeddings.
    """
    logger.debug("Starting PCA projection to 2 components.")
    logger.debug(
        f"Ref embeddings shape: {ref_embeddings.shape}, Synthetic embeddings shape: {synthetic_embeddings.shape}.")

    all_vecs = np.vstack([ref_embeddings, synthetic_embeddings])
    if all_vecs.shape[0] <= 1 or all_vecs.shape[1] < 2:
        logger.warning(
            f"Not enough samples or features for PCA (shape: {all_vecs.shape}). Returning original embeddings as is or zeros.")
        # If all_vecs.shape[1] is 1, PCA to 2 components will fail.
        # Handle by returning original or zeros.
        return np.zeros((ref_embeddings.shape[0], 2)), np.zeros(
            (synthetic_embeddings.shape[0], 2))  # Return zeros to avoid errors downstream

    try:
        pca = PCA(n_components=2, random_state=42)
        all_2d = pca.fit_transform(all_vecs)
        logger.debug(f"PCA fitted. Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
        logger.debug("PCA fitting and transformation complete.")
        return np.split(all_2d, [len(ref_embeddings)], axis=0)
    except Exception as e:
        logger.error(f"Error during PCA projection: {e}", exc_info=True)
        return np.zeros((ref_embeddings.shape[0], 2)), np.zeros((synthetic_embeddings.shape[0], 2))


def plot_2d_projection(ref_2d: np.ndarray, synthetic_2d: np.ndarray, colour: np.ndarray, save_to: str = None,
                       method_name: str = "PCA"):
    """
    Plots the 2D projection of embeddings.

    Args:
        ref_2d (np.ndarray): 2D projected reference embeddings.
        synthetic_2d (np.ndarray): 2D projected synthetic embeddings.
        colour (np.ndarray): Array of similarity scores for coloring synthetic points.
        save_to (str, optional): Path to save the plot. If None, only displays.
        method_name (str): Name of the projection method (e.g., "PCA", "t-SNE").
    """
    logger.info(f"Plotting 2D projection using {method_name}.")
    logger.debug(
        f"Ref 2D shape: {ref_2d.shape}, Synthetic 2D shape: {synthetic_2d.shape}, Color array shape: {colour.shape}.")

    if ref_2d.size == 0 and synthetic_2d.size == 0:
        logger.warning("No data to plot for 2D projection. Skipping plot.")
        return

    try:
        plt.figure(figsize=(6, 5))
        plt.scatter(ref_2d[:, 0], ref_2d[:, 1], alpha=.3, s=35, label="reference")

        # Only plot synthetic if data exists and is 2D
        if synthetic_2d.shape[0] > 0 and synthetic_2d.shape[1] == 2:
            sc = plt.scatter(synthetic_2d[:, 0], synthetic_2d[:, 1], c=colour, cmap="viridis",
                             marker="x", s=80, linewidths=2, label="synthetic frames")
            plt.colorbar(sc, label="max sim → reference")
        else:
            logger.warning("Synthetic 2D data is empty or not 2-dimensional. Skipping synthetic points in 2D plot.")

        plt.legend()
        plt.title(f"{method_name} of embeddings")
        plt.tight_layout()

        if save_to:
            try:
                plt.savefig(save_to, bbox_inches='tight', dpi=300)
                logger.info(f"2D {method_name} plot saved to {save_to}")
            except Exception as e:
                logger.error(f"Failed to save {method_name} plot to {save_to}: {e}", exc_info=True)
        else:
            logger.debug("2D projection plot not saved (save_to is None).")

        plt.show(block=False)  # Show plot without blocking execution
    except Exception as e:
        logger.error(f"Error generating 2D projection plot: {e}", exc_info=True)


def plot_similarity_matrix(tuning_cfg: TuningConfig, ref_embeddings: np.ndarray, synthetic_embeddings: np.ndarray,
                           max_labels: int = 30, save_to: str = None):
    """
    Plots a similarity matrix (heatmap) between reference and synthetic embeddings.

    Args:
        tuning_cfg (TuningConfig): The tuning configuration, specifying the similarity metric.
        ref_embeddings (np.ndarray): Embeddings from reference images.
        synthetic_embeddings (np.ndarray): Embeddings from synthetic images.
        max_labels (int): Maximum number of labels to display on the heatmap axes.
        save_to (str, optional): Path to save the plot.
    """
    logger.info(f"Plotting similarity matrix using metric '{tuning_cfg.similarity_metric}'.")
    logger.debug(
        f"Ref embeddings shape: {ref_embeddings.shape}, Synthetic embeddings shape: {synthetic_embeddings.shape}.")

    if ref_embeddings.size == 0 or synthetic_embeddings.size == 0:
        logger.warning("Reference or synthetic embeddings are empty. Skipping similarity matrix plot.")
        return

    all_vecs = np.vstack([ref_embeddings, synthetic_embeddings])
    num_vecs = len(all_vecs)
    sim_mat = np.zeros((num_vecs, num_vecs))

    precomputed_kwargs = {}
    try:
        precomputed_kwargs = precompute_matric_args(tuning_cfg, ref_embeddings)
        logger.debug("Pre-computed metric args for similarity matrix.")
    except Exception as e:
        logger.warning(
            f"Error during pre-computation for similarity matrix: {e}. Similarity will recompute args on-the-fly.",
            exc_info=True)

    logger.debug("Calculating pairwise similarity matrix (might be slow for large N)...")
    for a in tqdm(range(sim_mat.shape[0]), desc="Calculating similarity matrix rows"):
        for b in range(sim_mat.shape[1]):
            # The similarity function expects ref_embeddings to define the distribution
            # and synthetic_embeddings to be the one(s) being compared.
            # Here, we treat 'ref_embeddings' as the context for ALL comparisons.
            # This might need review if cross-comparison (synth vs synth) is intended
            # in a different context, but for now, it's comparing against the reference space.
            try:
                sim_mat[a, b] = similarity(
                    tuning_cfg=tuning_cfg,
                    ref_embeddings=ref_embeddings,  # Reference context
                    synthetic_embeddings=all_vecs[b].reshape(1, -1),  # Single vector being compared
                    **precomputed_kwargs,
                )
            except Exception as e:
                logger.error(f"Error computing similarity for matrix cell ({a},{b}): {e}. Setting to NaN.",
                             exc_info=True)
                sim_mat[a, b] = np.nan  # Mark failed computations

    labels = ([f"ref {idx}" for idx in range(len(ref_embeddings))] +
              [f"synth {idx}" for idx in range(len(synthetic_embeddings))])
    step = max(1, len(labels) // max_labels)
    logger.debug(f"Heatmap labels count: {len(labels)}, step for display: {step}.")

    try:
        plot_heatmap(sim_mat, labels, step, title=tuning_cfg.similarity_metric, save_to=save_to)
        logger.info(f"Similarity matrix heatmap saved to {save_to}")
    except Exception as e:
        logger.error(f"Failed to generate similarity heatmap: {e}", exc_info=True)


def plot_heatmap(matrix: np.ndarray, labels: List[str], step: int, *,
                 title: str, save_to: str = None):
    """
    Draw a square heat-map with readable tick labels.

    Args:
        matrix (np.ndarray): The 2D matrix to plot.
        labels (List[str]): Labels for the axes.
        step (int): Step size for displaying tick labels.
        title (str): Title of the heatmap.
        save_to (str, optional): Path to save the plot.
    """
    logger.debug(f"Plotting heatmap with title '{title}'. Matrix shape: {matrix.shape}.")

    if matrix.size == 0:
        logger.warning("Input matrix for heatmap is empty. Skipping plot.")
        return

    try:
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="viridis", interpolation='nearest')  # Added interpolation
        fig.colorbar(im, ax=ax, label=title)

        # ticks on top & left --------------------------------------------------
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        ticks_to_show = np.arange(0, n, step)
        ax.set_xticks(ticks_to_show)
        ax.set_yticks(ticks_to_show)

        x_labels = [labels[i] for i in ticks_to_show]
        y_labels = x_labels  # same order

        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(y_labels, fontsize=8)

        fig.tight_layout()

        if save_to:
            try:
                plt.savefig(save_to, bbox_inches='tight', dpi=300)
                logger.info(f"Heat-map saved to {save_to}")
            except Exception as e:
                logger.error(f"Failed to save heatmap to {save_to}: {e}", exc_info=True)
        else:
            logger.debug("Heatmap not saved (save_to is None).")

        plt.show(block=False)
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}", exc_info=True)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    Convert a 16-bit instance-ID mask into a BGR preview image.

    ID 0 → black, 1…N → HSV hues spaced evenly.
    """
    logger.debug(f"Converting mask of shape {mask.shape} and dtype {mask.dtype} to color image.")

    if mask.dtype != np.uint16:
        logger.debug(f"Converting mask from {mask.dtype} to np.uint16.")
        mask = mask.astype(np.uint16)

    max_id = mask.max()
    if max_id == 0:
        logger.warning("Mask contains only background (ID 0). Returning black image.")
        return np.zeros((*mask.shape, 3), dtype=np.uint8)

    hsv = np.zeros((*mask.shape, 3), dtype=np.uint8)
    hsv[..., 1:] = 255  # full saturation & value
    # Map IDs from 1 to max_id to hue values 0 to 179 (HSV hue range)
    # Ensure no division by zero if max_id is 0
    hsv[mask > 0, 0] = (mask[mask > 0].astype(np.float32) / max_id * 179).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[mask == 0] = 0  # Keep background black

    logger.debug("Mask converted to color BGR image.")
    return bgr


def show_frame(frame: np.ndarray, title: str = "") -> None:
    """
    Display a single frame using matplotlib.

    Args:
        frame (np.ndarray): The image frame to display, expected to be in RGB format for matplotlib.
        title (str): Title for the plot.
    """
    logger.info(f"Displaying frame: '{title}' (shape: {frame.shape}, dtype: {frame.dtype}).")

    try:
        plt.figure(figsize=(8, 6))  # Create a new figure
        if frame.ndim == 2:  # Grayscale image
            plt.imshow(frame, cmap='gray')
            logger.debug("Frame is grayscale.")
        elif frame.ndim == 3 and frame.shape[2] >= 3:  # Color image (RGB or BGR assumed to be RGB by imshow)
            # Ensure it's RGB for matplotlib. Assuming it came as BGR if from OpenCV.
            if frame.dtype == np.uint8 and frame.shape[2] == 3:  # Common case from OpenCV reads
                logger.debug("Frame is 3-channel, assuming BGR and converting to RGB for display.")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame)
            logger.debug("Frame is color.")
        else:
            msg = f"Frame must be a 2D or 3D numpy array, but got shape {frame.shape}."
            logger.error(msg)
            raise ValueError(msg)

        logger.debug(f"Frame max value: {frame.max()} ({frame.dtype}).")
        plt.title(title)
        plt.axis('off')  # Hide axes
        plt.tight_layout()  # Adjust layout
        plt.show(block=False)
        logger.info(f"Frame '{title}' displayed successfully.")
    except Exception as e:
        logger.error(f"Error displaying frame '{title}': {e}", exc_info=True)


def get_colormap(all_track_ids: set[int], cmap_name: str = 'tab20') -> dict[int, Tuple[int, int, int]]:
    """
    Generates a consistent color map for track IDs.

    Args:
        all_track_ids (set[int]): Set of all unique track IDs.
        cmap_name (str): Matplotlib colormap name.

    Returns:
        dict[int, Tuple[int, int, int]]: A dictionary mapping track IDs to RGB colors (0-255).
    """
    logger.debug(f"Generating colormap for {len(all_track_ids)} unique track IDs using '{cmap_name}'.")

    if not all_track_ids:
        logger.warning("No track IDs provided for colormap generation. Returning empty map.")
        return {}

    try:
        cmap = plt.get_cmap(cmap_name)
        num_colors = cmap.N  # Number of distinct colors in the colormap

        sorted_ids = sorted(list(all_track_ids))  # Convert set to list and sort for consistent assignment
        color_map = {
            track_id: tuple((np.array(cmap(i % num_colors)[:3]) * 255).astype(int))
            for i, track_id in enumerate(sorted_ids)
        }
        logger.debug("Colormap generated successfully.")
        return color_map
    except Exception as e:
        logger.error(f"Error generating colormap: {e}", exc_info=True)
        return {}


def visualize_tracked_masks(
        frames: List[np.ndarray],
        tracked_masks: List[np.ndarray],
        video_path: str,  # Path to original video (for naming output files)
        output_path: str,  # Base directory for outputs
        fps: float,
        alpha: float = 0.5,
):
    """
    Overlays tracked masks on frames and saves the result as an MP4 video,
    and optionally as a GIF, and generates kymographs.

    Args:
        frames (List[np.ndarray]): The original video frames (BGR).
        tracked_masks (List[np.ndarray]): The tracked segmentation masks (instance IDs).
        video_path (str): Original video path (used for naming output files).
        output_path (str): Base directory for saving all outputs (video, GIF, kymographs).
        fps (float): Frames per second for output videos/GIF.
        alpha (float): Transparency for mask overlay.
    """
    logger.info(f"Starting visualization of tracked masks for '{os.path.basename(video_path)}'.")
    logger.debug(
        f"Input frames: {len(frames)}, masks: {len(tracked_masks)}. Output path: {output_path}, FPS: {fps}, Alpha: {alpha}.")

    if not frames or not tracked_masks:
        logger.warning("Cannot visualize tracked masks: no frames or masks provided.")
        return
    if len(frames) != len(tracked_masks):
        logger.error(
            f"Number of frames ({len(frames)}) does not match number of masks ({len(tracked_masks)}). Skipping visualization.")
        return

    try: # <--- THIS IS THE CORRECTED OUTER TRY BLOCK for visualize_tracked_masks
        # --- 1. Create a consistent, random color map ---
        all_track_ids = {
            track_id for mask in tracked_masks for track_id in np.unique(mask) if track_id != 0
        }
        color_map = get_colormap(all_track_ids)
        logger.info(f"Generated colormap for {len(all_track_ids)} unique track IDs.")

        # --- 2. Prepare paths and writers ---
        base_output_path = Path(os.path.abspath(output_path))
        video_name = Path(video_path).stem  # e.g., "my_video" from "path/to/my_video.avi"

        try:
            base_output_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured base output path for visualizations exists: {base_output_path}.")
        except OSError as e:
            logger.error(f"Failed to create output directory {base_output_path}: {e}", exc_info=True)
            # Re-raise so the outer try can catch it and log a general failure.
            raise 

        video_output_path = base_output_path / (video_name + '_tracked.mp4')
        gif_output_path = base_output_path / (video_name + '_tracked.gif')
        kymograph_dir = base_output_path / (video_name + "_kymographs")
        logger.debug(f"Output paths: MP4='{video_output_path}', GIF='{gif_output_path}', Kymographs='{kymograph_dir}'.")

        if frames[0].ndim == 3:
            h, w, _ = frames[0].shape
        else:  # Grayscale input frame
            h, w = frames[0].shape
            logger.warning("Input frames are grayscale. Converting to BGR for visualization.")
            frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in frames]  # Convert to 3-channel for overlay

        # --- 3. Generate all visualized frames ---
        logger.info("Generating all visualized frames (overlaying masks, adding lengths)...")
        visualized_frames: List[np.ndarray] = []
        for frame_idx, (frame, mask) in enumerate(
                tqdm(zip(frames, tracked_masks), total=len(frames), desc="Overlaying Masks")):
            try:
                vis_frame = frame.copy()  # Make a copy to draw on
                overlay = vis_frame.copy()
                track_ids_in_frame = np.unique(mask)[1:]  # Exclude background ID 0

                for track_id in track_ids_in_frame:
                    color = color_map.get(track_id, (255, 255, 255))  # Default to white if ID not in map
                    instance_mask = (mask == track_id)

                    # Ensure color is a tuple for broadcasting if it came from something weird
                    color_tuple = tuple(color) if isinstance(color, (list, np.ndarray)) else color
                    overlay[instance_mask] = color_tuple  # Paint segmented area

                    # Calculate and display length
                    skeleton = skeletonize(instance_mask)
                    length = int(np.sum(skeleton))
                    logger.debug(f"Frame {frame_idx}, Track {track_id}: Skeleton length = {length}px.")

                    text_position = None
                    if length > 0:
                        # Attempt to find an endpoint for text
                        kernel = np.ones((3, 3), dtype=np.uint8)
                        neighbor_map = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
                        endpoints = np.argwhere((skeleton > 0) & (neighbor_map == 2))
                        if len(endpoints) > 0:
                            # Use the first endpoint for text position (x, y)
                            text_position = (endpoints[0][1], endpoints[0][0])
                            logger.debug(f"Frame {frame_idx}, Track {track_id}: Endpoint found at {text_position}.")
                        else:  # Fallback for loops or single points
                            coords = np.argwhere(instance_mask)
                            if len(coords) > 0:
                                # Use centroid for text position (x, y) if no endpoints
                                centroid_y, centroid_x = coords.mean(axis=0)
                                text_position = (int(centroid_x), int(centroid_y))
                                logger.debug(
                                    f"Frame {frame_idx}, Track {track_id}: No endpoint, using centroid at {text_position}.")
                            else:
                                logger.warning(
                                    f"Frame {frame_idx}, Track {track_id}: No coordinates found for text positioning.")

                    if text_position:
                        text = f"{length}px"
                        # Adjust position slightly for better visibility
                        pos = (text_position[0] + 8, text_position[1] + 8)
                        # Draw text with outline for readability
                        cv2.putText(overlay, text, (pos[0] + 1, pos[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                                    2, cv2.LINE_AA)
                        cv2.putText(overlay, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        logger.debug(f"Frame {frame_idx}, Track {track_id}: Added length text '{text}' at {pos}.")

                # Blend the overlay and store the final frame
                cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)
                visualized_frames.append(vis_frame)
            except Exception as e:
                logger.error(f"Error processing frame {frame_idx} for tracked mask visualization: {e}", exc_info=True)
                # If an error occurs, append the original frame to avoid breaking the video
                visualized_frames.append(frame.copy())

        # --- 4. Save the generated frames to files ---
        logger.info("Saving output files (MP4, GIF)...")

        # Save MP4 video
        if visualized_frames:
            try:
                video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps), (w, h))
                if not video_writer.isOpened():
                    raise IOError("OpenCV VideoWriter failed to open.")

                for vis_frame in tqdm(visualized_frames, desc="Saving MP4"):
                    video_writer.write(vis_frame)
                video_writer.release()
                logger.info(f"Successfully saved visualized video to: {video_output_path}")
            except Exception as e:
                logger.error(f"Failed to save MP4 video to {video_output_path}: {e}", exc_info=True)
        else:
            logger.warning("No visualized frames to save as MP4. Skipping MP4 export.")

        # Save GIF
        if visualized_frames:
            try:
                # IMPORTANT: Convert BGR frames from OpenCV to RGB for imageio
                rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in visualized_frames]

                # The `duration` parameter in imageio is in seconds per frame
                duration = 1 / fps if fps > 0 else 0.1
                if duration <= 0:
                    duration = 0.1  # Prevent invalid duration if fps is very high or zero
                    logger.warning(f"Adjusted GIF frame duration to {duration}s as FPS was {fps}.")

                imageio.mimsave(str(gif_output_path), rgb_frames, duration=duration)
                logger.info(f"Successfully saved visualized GIF to: {gif_output_path}")
            except Exception as e:
                logger.error(f"Failed to save GIF to {gif_output_path}: {e}", exc_info=True)
        else:
            logger.warning("No visualized frames to save as GIF. Skipping GIF export.")

        logger.info("Generating kymographs...")
        try:
            kymograph_dir.mkdir(parents=True, exist_ok=True)  # Ensure kymograph directory exists
            generate_kymographs(frames, tracked_masks, str(kymograph_dir))  # Pass original frames
            logger.info(f"Kymographs generated and saved to: {kymograph_dir}")
        except Exception as e:
            logger.error(f"Failed to generate kymographs: {e}", exc_info=True)

        logger.info(f"Tracked masks visualization complete for '{os.path.basename(video_path)}'.")

    except Exception as e: # <--- THIS IS THE NEWLY ADDED EXCEPT BLOCK that balances the outer try
        logger.error(f"An unexpected error occurred in visualize_tracked_masks for '{os.path.basename(video_path)}': {e}", exc_info=True)