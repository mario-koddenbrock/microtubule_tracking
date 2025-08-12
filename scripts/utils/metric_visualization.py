import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from microtubule_tracking.config.tuning import TuningConfig
from microtubule_tracking.data_generation.optimization.embeddings import ImageEmbeddingExtractor
from microtubule_tracking.data_generation.optimization.metrics import precompute_matric_args, similarity
from microtubule_tracking.data_generation.optimization.toy_data import get_toy_data

logger = logging.getLogger(__name__)


def load_frames_from_videos(video_dir: Path, num_frames_per_video: int = 5) -> List[np.ndarray]:
    """Loads a specified number of frames from each video in a directory."""
    images = []
    video_files = sorted(list(video_dir.glob("*.mp4")))
    if not video_files:
        logger.warning(f"No video files found in '{video_dir}'.")
        return images

    logger.debug(f"Loading frames from {len(video_files)} videos in '{video_dir}'...")
    for video_file in video_files:
        cap = cv2.VideoCapture(str(video_file))
        count = 0
        while cap.isOpened() and count < num_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
        cap.release()
    return images


def load_images_from_dir(image_dir: Path) -> List[np.ndarray]:
    """Loads all images from a directory."""
    images = []
    image_files = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    if not image_files:
        logger.warning(f"No image files found in '{image_dir}'.")
        return images

    logger.debug(f"Loading {len(image_files)} images from '{image_dir}'...")
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(np.array(img))
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
    return images


def calculate_similarity_scores(
    cfg: TuningConfig,
    ref_embeddings: np.ndarray,
    target_embeddings: np.ndarray
) -> np.ndarray:
    """Calculates similarity scores for target embeddings against reference embeddings."""
    if target_embeddings is None or target_embeddings.shape[0] == 0:
        return np.array([])

    is_lpips_model = "lpips" in cfg.model_name
    precomputed_kwargs = {}
    if not is_lpips_model:
        precomputed_kwargs = precompute_matric_args(cfg, ref_embeddings)

    # KID is calculated over the whole batch, others are per-image
    if cfg.similarity_metric == "kid":
        score = similarity(
            tuning_cfg=cfg,
            ref_embeddings=ref_embeddings,
            synthetic_embeddings=target_embeddings,
            **precomputed_kwargs,
        )
        return np.array([score])

    if is_lpips_model:
        synthetic_embeddings=lambda i: target_embeddings[i, :]
    else:
        synthetic_embeddings=lambda i: target_embeddings[i, :].reshape(1, -1),

    scores = [
        similarity(
            tuning_cfg=cfg,
            ref_embeddings=ref_embeddings,
            synthetic_embeddings=synthetic_embeddings(i),
            **precomputed_kwargs,
        ) for i in range(target_embeddings.shape[0])
    ]
    return np.array(scores)


def preprocess_image_for_plot(img: np.ndarray, size: int = 96) -> np.ndarray:
    """Crops an image to a square and resizes it for plotting."""
    h, w, _ = img.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    return cv2.resize(cropped_img, (size, size), interpolation=cv2.INTER_AREA)


def plot_scores_with_images(
    scores: Dict[str, np.ndarray],
    images: Dict[str, Optional[np.ndarray]],
    output_path: Path,
    model_name: str,
    metric_name: str,
    layer_name: str
):
    """Generates and saves a Seaborn box plot of scores with example images."""
    valid_labels = [label for label, s in scores.items() if s is not None and len(s) > 0]
    if not valid_labels:
        logger.warning(f"No score data to plot for {model_name}/{metric_name}/layer_{layer_name}.")
        return

    plot_data = [{'Data Source': label, 'Similarity Score': score}
                 for label in valid_labels for score in scores[label]]
    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.boxplot(x='Data Source', y='Similarity Score', data=df, ax=ax, order=valid_labels,
                palette="Set2", boxprops=dict(alpha=.8), showfliers=False,
                hue='Data Source', legend=False)
    sns.stripplot(x='Data Source', y='Similarity Score', data=df, ax=ax, order=valid_labels,
                  color=".25", size=3, jitter=True)

    ax.set_xlabel(None)
    title = f"Metric: {metric_name.upper()} | Model: {model_name.split('/')[-1]} | Layer: {layer_name}"
    ax.set_title(title, fontsize=16)

    if metric_name == "cosine":
        ax.set_ylim(0, 1)

    fig.subplots_adjust(bottom=0.25)

    medians = df.groupby('Data Source')['Similarity Score'].median().reindex(valid_labels)
    for i, label in enumerate(valid_labels):
        median_value = medians[label]
        ax.text(i, median_value, f'{median_value:.3f}', va='center', ha='center',
                color='white', weight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.7))

    for i, label in enumerate(valid_labels):
        if images.get(label) is not None:
            img = preprocess_image_for_plot(images[label])
            imagebox = OffsetImage(img, zoom=0.75)
            ab = AnnotationBbox(imagebox, (i, 0), xybox=(0., -60.), frameon=False,
                                xycoords=('data', 'axes fraction'), boxcoords="offset points", pad=0)
            ax.add_artist(ab)

    plt.savefig(output_path, bbox_inches='tight')
    logger.info(f"Box plot saved to '{output_path}'")
    plt.close(fig)



def load_all_data(cfg: TuningConfig) -> Dict[str, List[np.ndarray]]:
    """Loads all image datasets required for the comparison."""
    logger.info("--- Loading All Image Data ---")
    reference_images = load_frames_from_videos(Path(cfg.reference_series_dir))
    manual_images = load_images_from_dir(Path("data/synthetic_manual"))
    optimized_images = load_images_from_dir(Path("data/optimization/config_B"))
    toy_data: Dict[str, Any] = get_toy_data()
    toy_images = toy_data.get("images", [])

    all_image_data = {
        "Reference": reference_images,
        "Manual": manual_images,
        "Optimized": optimized_images,
        "Toy": toy_images
    }
    for name, data in all_image_data.items():
        logger.info(f"Loaded {len(data)} images for '{name}' dataset.")
    return all_image_data


def get_data_for_model(
    cfg: TuningConfig,
    all_image_data: Dict[str, List[np.ndarray]],
    is_lpips: bool
) -> Dict[str, np.ndarray]:
    """
    Returns embeddings for standard models or raw images for LPIPS models.
    """
    if is_lpips:
        logger.debug("Using raw images as data for LPIPS model.")
        return {label: np.array(images) for label, images in all_image_data.items()}

    logger.debug(f"Computing embeddings for model: {cfg.model_name}")
    embedding_extractor = ImageEmbeddingExtractor(cfg)
    return {
        "Reference": embedding_extractor.extract_from_references(),
        "Manual": embedding_extractor.extract_from_frames(all_image_data["Manual"], len(all_image_data["Manual"])),
        "Optimized": embedding_extractor.extract_from_frames(all_image_data["Optimized"], len(all_image_data["Optimized"])),
        "Toy": embedding_extractor.extract_from_frames(all_image_data["Toy"], len(all_image_data["Toy"]))
    }


def calculate_all_scores(
    cfg: TuningConfig,
    all_embeddings: Dict[str, np.ndarray],
    metrics_to_test: List[str]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Calculates similarity scores for all metrics."""
    all_metric_scores = {}
    ref_data = all_embeddings.get("Reference")

    if ref_data is None or ref_data.shape[0] == 0:
        logger.warning("Reference data is empty, skipping score calculation.")
        return {}

    for metric_name in metrics_to_test:
        logger.debug(f"Calculating scores for metric: {metric_name.upper()}")
        cfg.similarity_metric = metric_name
        scores_for_metric = {}
        for label, target_data in all_embeddings.items():
            if target_data is None or target_data.shape[0] == 0:
                scores_for_metric[label] = np.array([])
                continue
            scores_for_metric[label] = calculate_similarity_scores(
                cfg=cfg,
                ref_embeddings=ref_data,
                target_embeddings=target_data
            )
        all_metric_scores[metric_name] = scores_for_metric
    return all_metric_scores


def process_model_evaluation(
    cfg: TuningConfig,
    model_name: str,
    layer_indices: List[int],
    metrics_to_test: List[str],
    all_image_data: Dict[str, List[np.ndarray]],
    example_images: Dict[str, np.ndarray],
    output_dir: Path
):
    """Processes a single model across all its layers and specified metrics."""
    logger.info(f"\n{'='*20} Processing Model: {model_name} {'='*20}")
    cfg.model_name = model_name
    is_lpips_model = "lpips" in model_name
    metrics_for_model = [model_name] if is_lpips_model else metrics_to_test

    for layer_idx in layer_indices:
        logger.info(f"--- Processing Layer: {layer_idx} ---")
        cfg.embedding_layer = layer_idx

        # Get embeddings or raw images
        all_data = get_data_for_model(cfg, all_image_data, is_lpips_model)

        # Calculate scores for all metrics
        all_scores_by_metric = calculate_all_scores(cfg, all_data, metrics_for_model)

        # Plot results for each metric
        for metric_name, scores in all_scores_by_metric.items():
            model_filename_part = model_name.replace('/', '_')
            layer_name = 'img' if is_lpips_model else str(layer_idx)
            plot_filename = f"metrics_{model_filename_part}_{metric_name}_layer_{layer_name}.png"
            plot_path = output_dir / plot_filename

            plot_scores_with_images(
                scores=scores,
                images=example_images,
                output_path=plot_path,
                model_name=cfg.model_name,
                metric_name=metric_name,
                layer_name=layer_name
            )
