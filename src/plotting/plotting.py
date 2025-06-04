import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


def visualize_embeddings(best_cfg, ref_vecs, best_vecs, output_dir:str = "plots/",
                         perplexity: int = 30, metric: str = "cosine", max_labels: int = 30):
    """Render both the heat‑map *and* the t‑SNE plot.

    Parameters
    ----------
    best_cfg : SyntheticDataConfig
    ref_vecs : np.ndarray
    best_vecs : np.ndarray
    output_dir : str, optional – directory to save the plots
    perplexity : int, optional – t‑SNE perplexity
    metric : str, optional – distance metric for t‑SNE
    max_labels : int, optional – maximum tick labels on the heat‑map axes
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1 – Heat‑map -----------------------------------------------------------
    heatmap_path = f"{output_dir}/heatmap_{best_cfg.id}.png"
    plot_similarity_matrix(ref_vecs, best_vecs, max_labels=max_labels, save_to=heatmap_path)

    # 2 – t‑SNE -------------------------------------------------------------
    tsne_path = f"{output_dir}/tsne_{best_cfg.id}.png"
    ref_2d, best_2d = tsne_projection(ref_vecs, best_vecs, metric=metric, perplexity=perplexity)
    colour = cosine_similarity(best_vecs, ref_vecs).max(axis=1)  # (N_best,)
    plot_tsne(ref_2d, best_2d, colour, save_to=tsne_path)

def tsne_projection(ref_vecs: np.ndarray, best_vecs: np.ndarray, *,
                     metric: str, perplexity: int):
    all_vecs = np.vstack([ref_vecs, best_vecs])
    perplexity = min(perplexity, ref_vecs.shape[0])
    tsne = TSNE(n_components=2, metric=metric, perplexity=perplexity,
                init="pca", random_state=42)
    all_2d = tsne.fit_transform(all_vecs)
    return np.split(all_2d, [len(ref_vecs)], axis=0)


def plot_tsne(ref_2d: np.ndarray, best_2d: np.ndarray, colour: np.ndarray, save_to: str = None):
    plt.figure(figsize=(6, 5))
    plt.scatter(ref_2d[:, 0], ref_2d[:, 1], alpha=.3, s=35, label="reference")
    sc = plt.scatter(best_2d[:, 0], best_2d[:, 1], c=colour, cmap="viridis",
                     marker="x", s=80, linewidths=2, label="synthetic frames")
    plt.colorbar(sc, label="mean cosine sim → reference")
    plt.legend()
    plt.title("t‑SNE of embeddings")
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, bbox_inches='tight', dpi=300)
        print(f"✓ t‑SNE plot saved to {save_to}")

    plt.show()


def plot_similarity_matrix(ref_vecs: np.ndarray, best_vecs: np.ndarray, *,
                            max_labels: int, save_to:str = None):
    """Plot a square cosine‑similarity matrix for `[ref … synthetic_best]`."""
    best_vecs = best_vecs.mean(axis=0, keepdims=True)      # (1, D)
    all_vecs  = np.vstack([ref_vecs, best_vecs])           # (N_ref+1, D)
    sim_mat   = cosine_similarity(all_vecs, all_vecs)

    labels = ([f"ref {idx}" for idx in range(len(ref_vecs))] +
              [f"best {idx}" for idx in range(len(best_vecs))])
    step   = max(1, len(labels) // max_labels)

    plot_heatmap(sim_mat, labels, step, title="Reference + best‑synthetic", save_to=save_to)

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
    fig.colorbar(im, ax=ax, label="cosine similarity")

    # ticks on top & left --------------------------------------------------
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ticks_to_show = np.arange(0, n, step)
    ax.set_xticks(ticks_to_show)
    ax.set_yticks(ticks_to_show)

    x_labels = [labels[i] for i in ticks_to_show]
    y_labels = x_labels  # same order

    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8)

    ax.set_title(title)
    fig.tight_layout()

    if save_to:
        plt.savefig(save_to, bbox_inches='tight', dpi=300)
        print(f"✓ Heat-map saved to {save_to}")

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
    hsv[..., 1:] = 255                          # full saturation & value
    hsv[..., 0]  = (mask.astype(np.float32) / max_id * 179).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[mask == 0] = 0
    return bgr