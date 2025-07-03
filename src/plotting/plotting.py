import os
from pathlib import Path
from typing import List

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from analysis.kymographs import generate_kymographs
from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation.optimization.metrics import similarity


def visualize_embeddings(cfg:SyntheticDataConfig, tuning_cfg:TuningConfig,
                         ref_embeddings, synthetic_embeddings, output_dir: str = "plots/"):

    os.makedirs(output_dir, exist_ok=True)

    heatmap_path = f"{output_dir}/heatmap_{cfg.id}.png"
    plot_similarity_matrix(tuning_cfg, ref_embeddings, synthetic_embeddings, save_to=heatmap_path)

    colour = np.array([similarity(
        tuning_cfg=tuning_cfg,
        ref_embeddings=ref_embeddings,
        synthetic_embeddings=synthetic_embeddings[i, :].reshape(1, -1),
    ) for i in range(synthetic_embeddings.shape[0])])

    # ref_2d, synthetic_2d = tsne_projection(ref_embeddings, synthetic_embeddings, similarity_metric=tuning_cfg.similarity_metric, perplexity=30)
    ref_2d, synthetic_2d = pca_projection(ref_embeddings, synthetic_embeddings)
    projection_path = f"{output_dir}/tsne_{cfg.id}.png"

    plot_2d_projection(ref_2d, synthetic_2d, colour, save_to=projection_path)


def tsne_projection(ref_embeddings: np.ndarray, synthetic_embeddings: np.ndarray, similarity_metric: str, perplexity: int):
    all_vecs = np.vstack([ref_embeddings, synthetic_embeddings])
    perplexity = min(perplexity, ref_embeddings.shape[0])
    tsne = TSNE(n_components=2, metric=similarity_metric, perplexity=perplexity, init="pca", random_state=42)
    all_2d = tsne.fit_transform(all_vecs)
    return np.split(all_2d, [len(ref_embeddings)], axis=0)


def pca_projection(ref_embeddings: np.ndarray, synthetic_embeddings: np.ndarray):
    all_vecs = np.vstack([ref_embeddings, synthetic_embeddings])
    pca = PCA(n_components=2, random_state=42)
    all_2d = pca.fit_transform(all_vecs)
    return np.split(all_2d, [len(ref_embeddings)], axis=0)

def plot_2d_projection(ref_2d: np.ndarray, synthetic_2d: np.ndarray, colour: np.ndarray, save_to: str = None, method_name: str = "PCA"):
    plt.figure(figsize=(6, 5))
    plt.scatter(ref_2d[:, 0], ref_2d[:, 1], alpha=.3, s=35, label="reference")
    sc = plt.scatter(synthetic_2d[:, 0], synthetic_2d[:, 1], c=colour, cmap="viridis",
                     marker="x", s=80, linewidths=2, label="synthetic frames")
    plt.colorbar(sc, label="max sim → reference")
    plt.legend()
    plt.title(f"{method_name} of embeddings")
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, bbox_inches='tight', dpi=300)
        print(f"{method_name} plot saved to {save_to}")

    plt.show()


def plot_similarity_matrix(tuning_cfg:TuningConfig, ref_embeddings: np.ndarray, synthetic_embeddings: np.ndarray,
                           max_labels: int = 30, save_to: str = None):

    # synthetic_embeddings = synthetic_embeddings.mean(axis=0, keepdims=True)
    all_vecs = np.vstack([ref_embeddings, synthetic_embeddings])
    num_vecs = len(all_vecs)
    sim_mat = np.zeros((num_vecs, num_vecs))

    for a in range(sim_mat.shape[0]):
        for b in range(sim_mat.shape[1]):
            sim_mat[a, b] = similarity(
                tuning_cfg=tuning_cfg,
                ref_embeddings=all_vecs[a].reshape(1, -1),
                synthetic_embeddings=all_vecs[b].reshape(1, -1)
            )

    labels = ([f"ref {idx}" for idx in range(len(ref_embeddings))] +
              [f"synth {idx}" for idx in range(len(synthetic_embeddings))])
    step = max(1, len(labels) // max_labels)

    plot_heatmap(sim_mat, labels, step, title=tuning_cfg.similarity_metric, save_to=save_to)


def plot_heatmap(matrix: np.ndarray, labels: list[str], step: int, *,
                 title: str, save_to: str = None):
    """Draw a square heat‑map with readable tick labels.

    * X‑axis labels appear on the top; long labels are rotated 45°
      and right‑aligned so they don’t overlap.
    * Y‑axis labels stay horizontal for easier reading.
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    # im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    im = ax.imshow(matrix, cmap="viridis")
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

    # ax.set_title(title)
    fig.tight_layout()

    if save_to:
        plt.savefig(save_to, bbox_inches='tight', dpi=300)
        print(f"Heat-map saved to {save_to}")

    plt.show()


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    Convert a 16-bit instance-ID mask into a BGR preview image.

    ID 0 → black, 1…N → HSV hues spaced evenly.
    """
    if mask.dtype != np.uint16:
        mask = mask.astype(np.uint16)

    max_id = mask.max()
    if max_id == 0:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)

    hsv = np.zeros((*mask.shape, 3), dtype=np.uint8)
    hsv[..., 1:] = 255  # full saturation & value
    hsv[..., 0] = (mask.astype(np.float32) / max_id * 179).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[mask == 0] = 0
    return bgr


def show_frame(frame: np.ndarray, title: str = "") -> None:
    """
    Display a single frame using matplotlib.

    Parameters
    ----------
    frame : np.ndarray
        The image frame to display, expected to be in BGR format.
        :param frame:
        :param title:
    """
    if frame.ndim == 2:  # Grayscale image
        plt.imshow(frame, cmap='gray')
    elif frame.ndim == 3:  # Color image
        plt.imshow(frame)
    else:
        raise ValueError("Frame must be a 2D or 3D numpy array.")

    print(f"{title}: {frame.max()} ({frame.dtype})")
    plt.title(title)
    plt.axis('off')  # Hide axes
    plt.show()




def get_colormap(all_track_ids, cmap_name='tab20'):
    cmap = plt.get_cmap(cmap_name)
    num_colors = cmap.N  # Number of distinct colors in the colormap

    sorted_ids = sorted(all_track_ids)  # To ensure consistent color assignment
    color_map = {
        track_id: tuple((np.array(cmap(i % num_colors)[:3]) * 255).astype(int))
        for i, track_id in enumerate(sorted_ids)
    }
    return color_map



def visualize_tracked_masks(
        frames: List[np.ndarray],
        tracked_masks: List[np.ndarray],
        video_path: str,
        output_path: str,
        fps: float,
        alpha: float = 0.5,
):
    """
    Overlays tracked masks on frames and saves the result as an MP4 video,
    and optionally as a GIF.

    This version:
    1.  Paints the segmented area with a transparent color.
    2.  Keeps colors consistent for each object over time.
    3.  Calculates and displays the length of each object.
    4.  Can export both MP4 and GIF formats from the same generated frames.
    """
    if not frames or not tracked_masks:
        print("Cannot visualize, no frames or masks provided.")
        return

    # --- 1. Create a consistent, random color map ---
    all_track_ids = {
        track_id for mask in tracked_masks for track_id in np.unique(mask) if track_id != 0
    }
    color_map = get_colormap(all_track_ids)

    # --- 2. Prepare paths and writers ---
    base_path = Path(os.path.abspath(output_path))
    video_name = Path(video_path).stem
    video_output_path = base_path / (video_name + '.mp4')
    gif_output_path = base_path / (video_name + '.gif')
    kymograph_dir = base_path / (video_name + "_kymographs")

    h, w = frames[0].shape[:2]

    # --- 3. Generate all visualized frames ---
    print("Generating visualized frames...")
    visualized_frames = []
    for frame, mask in tqdm(zip(frames, tracked_masks), total=len(frames), desc="Generating Frames"):
        vis_frame = frame.copy()
        overlay = vis_frame.copy()
        track_ids_in_frame = np.unique(mask)[1:]

        for track_id in track_ids_in_frame:
            color = color_map.get(track_id, (255, 255, 255))
            instance_mask = (mask == track_id)
            overlay[instance_mask] = color

            skeleton = skeletonize(instance_mask)
            length = int(np.sum(skeleton))

            text_position = None
            if length > 0:
                kernel = np.ones((3, 3), dtype=np.uint8)
                neighbor_map = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
                endpoints = np.argwhere((skeleton > 0) & (neighbor_map == 2))
                if len(endpoints) > 0:
                    text_position = (endpoints[0][1], endpoints[0][0])

            if text_position is None:
                coords = np.argwhere(instance_mask)
                if len(coords) > 0:
                    text_position = (int(coords.mean(axis=1)[1]), int(coords.mean(axis=0)[0]))

            if text_position:
                text = f"{length}px"
                pos = (text_position[0] + 8, text_position[1] + 8)
                cv2.putText(overlay, text, (pos[0] + 1, pos[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                            cv2.LINE_AA)
                cv2.putText(overlay, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend the overlay and store the final frame
        cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)
        visualized_frames.append(vis_frame)

    # --- 4. Save the generated frames to files ---
    print("\nSaving output files...")
    # Save MP4 video
    video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps), (w, h))
    for vis_frame in tqdm(visualized_frames, desc="Saving MP4"):
        video_writer.write(vis_frame)
    video_writer.release()
    print(f"Successfully saved visualized video to: {video_output_path}")

    # Save GIF
    # IMPORTANT: Convert BGR frames from OpenCV to RGB for imageio
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in visualized_frames]

    # The `duration` parameter in imageio is in seconds per frame
    duration = 1 / fps if fps > 0 else 0.1

    imageio.mimsave(gif_output_path, rgb_frames, duration=duration)
    print(f"Successfully saved visualized GIF to: {gif_output_path}")

    print("Generating kymographs...")
    generate_kymographs(frames, tracked_masks, kymograph_dir)
