import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from data_generation.utils import flatten_embeddings, cfg_to_embeddings


def visualize_embeddings(best_cfg, model, extractor, ref_embeddings,
                         *, perplexity: int = 30, metric: str = "cosine",
                         max_labels: int = 30):
    """Render both the heat‑map *and* the t‑SNE plot.

    Parameters
    ----------
    best_cfg : SyntheticDataConfig
    model, extractor : callable by `compute_embedding`
    ref_embeddings : array‑like, (N_ref, H, W) or (N_ref, D)
    perplexity : int, optional – t‑SNE perplexity
    metric : str, optional – distance metric for t‑SNE
    max_labels : int, optional – maximum tick labels on the heat‑map axes
    """
    # 0 – Prepare flat reference + per‑frame synthetic embeddings ------------
    ref_vecs  = flatten_embeddings(ref_embeddings)              # (N_ref, D)
    best_vecs = cfg_to_embeddings(best_cfg, model, extractor)   # (N_best, D)

    # 1 – Heat‑map -----------------------------------------------------------
    plot_similarity_matrix(ref_vecs, best_vecs, max_labels=max_labels)

    # 2 – t‑SNE -------------------------------------------------------------
    ref_2d, best_2d = tsne_projection(ref_vecs, best_vecs,
                                       metric=metric, perplexity=perplexity)

    colour = cosine_similarity(best_vecs, ref_vecs).max(axis=1)  # (N_best,)
    plot_tsne(ref_2d, best_2d, colour)

def tsne_projection(ref_vecs: np.ndarray, best_vecs: np.ndarray, *,
                     metric: str, perplexity: int):
    all_vecs = np.vstack([ref_vecs, best_vecs])
    tsne = TSNE(n_components=2, metric=metric, perplexity=perplexity,
                init="pca", random_state=0)
    all_2d = tsne.fit_transform(all_vecs)
    return np.split(all_2d, [len(ref_vecs)], axis=0)


def plot_tsne(ref_2d: np.ndarray, best_2d: np.ndarray, colour: np.ndarray):
    plt.figure(figsize=(6, 5))
    plt.scatter(ref_2d[:, 0], ref_2d[:, 1], alpha=.3, s=35, label="reference")
    sc = plt.scatter(best_2d[:, 0], best_2d[:, 1], c=colour, cmap="viridis",
                     marker="x", s=80, linewidths=2, label="synthetic frames")
    plt.colorbar(sc, label="mean cosine sim → reference")
    plt.legend()
    plt.title("t‑SNE of embeddings")
    plt.tight_layout()
    plt.show()


def plot_similarity_matrix(ref_vecs: np.ndarray, best_vecs: np.ndarray, *,
                            max_labels: int):
    """Plot a square cosine‑similarity matrix for `[ref … synthetic_best]`."""
    best_vecs = best_vecs.mean(axis=0, keepdims=True)      # (1, D)
    all_vecs  = np.vstack([ref_vecs, best_vecs])           # (N_ref+1, D)
    sim_mat   = cosine_similarity(all_vecs, all_vecs)

    labels = ([f"ref {idx}" for idx in range(len(ref_vecs))] +
              [f"best {idx}" for idx in range(len(best_vecs))])
    step   = max(1, len(labels) // max_labels)

    plot_heatmap(sim_mat, labels, step, title="Reference + best‑synthetic")

def plot_heatmap(matrix: np.ndarray, labels: list[str], step: int, *,
                  title: str):
    """Draw a square heat‑map with readable tick labels.

    * X‑axis labels appear on the top; long labels are rotated 45°
      and right‑aligned so they don’t overlap.
    * Y‑axis labels stay horizontal for easier reading.
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=(6, 5))
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
    plt.show()
