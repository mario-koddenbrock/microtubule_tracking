import logging
import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from config.tuning import TuningConfig
from data_generation.optimization.embeddings import ImageEmbeddingExtractor
from data_generation.optimization.metrics import precompute_matric_args, similarity
from data_generation.optimization.toy_data import get_toy_data
from scripts.utils.cli import parse_optimization_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_frames_from_videos(video_dir: Path, num_frames_per_video: int = 5) -> list[np.ndarray]:
    """Loads a specified number of frames from each video in a directory."""
    images = []
    video_files = sorted(list(video_dir.glob("*.mp4")))
    if not video_files:
        logger.warning(f"No video files found in '{video_dir}'.")
        return images
    logger.info(f"Loading frames from {len(video_files)} videos in '{video_dir}'...")
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


def load_images_from_dir(image_dir: Path) -> list[np.ndarray]:
    """Loads all images from a directory."""
    images = []
    image_files = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    if not image_files:
        logger.warning(f"No image files found in '{image_dir}'.")
        return images
    logger.info(f"Loading {len(image_files)} images from '{image_dir}'...")
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(np.array(img))
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
    return images


def calculate_similarity_scores(cfg: TuningConfig, ref_embeddings: np.ndarray,
                                target_embeddings: np.ndarray) -> np.ndarray:
    """Calculates similarity scores for target embeddings against reference embeddings."""
    if target_embeddings is None or target_embeddings.shape[0] == 0:
        return np.array([])

    precomputed_kwargs = precompute_matric_args(cfg, ref_embeddings)
    scores = np.array([similarity(
        tuning_cfg=cfg,
        ref_embeddings=ref_embeddings,
        synthetic_embeddings=target_embeddings[i, :].reshape(1, -1),
        **precomputed_kwargs,
    ) for i in range(target_embeddings.shape[0])])

    scores[np.isinf(scores)] = 0
    return scores


def preprocess_image_for_plot(img: np.ndarray, size: int = 96) -> np.ndarray:
    """Crops an image to a square and resizes it for plotting."""
    h, w, _ = img.shape
    min_dim = min(h, w)

    # Center crop to a square
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]

    # Scale down
    return cv2.resize(cropped_img, (size, size), interpolation=cv2.INTER_AREA)


def plot_scores_with_images(scores: Dict[str, np.ndarray], images: Dict[str, np.ndarray], output_path: Path,
                            model_name: str, metric_name: str, layer_name: str):
    """Generates and saves a Seaborn box plot of scores with a preprocessed example image and median value."""
    valid_labels = [label for label, s in scores.items() if s is not None and len(s) > 0]
    if not valid_labels:
        logger.warning("No score data to plot.")
        return

    # Prepare data for Seaborn: long-form DataFrame
    plot_data = []
    for label in valid_labels:
        for score in scores[label]:
            plot_data.append({'Data Source': label, 'Similarity Score': score})
    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Create box plot using Seaborn
    sns.boxplot(x='Data Source', y='Similarity Score', data=df, ax=ax, order=valid_labels,
                palette="Set2", boxprops=dict(alpha=.8), showfliers=False,
                hue='Data Source', legend=False)
    sns.stripplot(x='Data Source', y='Similarity Score', data=df, ax=ax, order=valid_labels,
                  color=".25", size=3, jitter=True)

    ax.set_xlabel(None)
    # Create a descriptive title
    title = f"Metric: {metric_name.upper()} | Model: {model_name.split('/')[-1]} | Layer: {layer_name}"
    ax.set_title(title)

    # Set y-axis limits for cosine similarity for better comparison
    if metric_name == "cosine":
        ax.set_ylim(0, 1)

    # Adjust layout to make space for images at the bottom
    fig.subplots_adjust(bottom=0.25)

    # Calculate medians and add text to each box
    medians = df.groupby('Data Source')['Similarity Score'].median().reindex(valid_labels)
    for i, label in enumerate(valid_labels):
        median_value = medians[label]
        ax.text(i, median_value, f'{median_value:.3f}',
                va='center', ha='center', color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.7))

    # Overlay a preprocessed example image for each category below the x-axis
    for i, label in enumerate(valid_labels):
        if label in images and images[label] is not None:
            img = preprocess_image_for_plot(images[label])
            imagebox = OffsetImage(img, zoom=0.75)

            ab = AnnotationBbox(imagebox, (i, 0),
                                xybox=(0., -60.),  # Offset in points
                                frameon=False,
                                xycoords=('data', 'axes fraction'),
                                boxcoords="offset points",
                                pad=0)
            ax.add_artist(ab)

    plt.savefig(output_path)
    logger.info(f"Box plot saved to '{output_path}'")
    plt.close(fig)  # Close the figure to free up memory


def main():
    """
    Main script to load data, compute embeddings for different models and layers,
    calculate similarity scores for various metrics, and plot the results.
    """
    _, _, config_path = parse_optimization_args()

    try:
        cfg = TuningConfig.load(config_path)
    except Exception as e:
        print(f"Error: Failed to load or parse the config file '{config_path}'.")
        print(f"Details: {e}")
        sys.exit(1)

    # --- Define models, metrics, and layer indices to evaluate ---
    models_to_test = ["lpips-alex", "lpips-vgg", "openai/clip-vit-base-patch32", "facebook/dinov2-large"]
    metrics_to_test = ["cosine", "fid", "kid", "jsd", "mahalanobis"]
    layer_indices_map = {
        "openai/clip-vit-base-patch32": list(range(1, 13)),
        "facebook/dinov2-large": list(range(1, 25)),
        "lpips-alex": [0],  # Operates on images, use dummy layer 0
        "lpips-vgg": [0],  # Operates on images, use dummy layer 0
    }

    logger.info(f"Will generate plots for models: {models_to_test}")

    # --- Create output directory ---
    output_dir = Path("plots/metric_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load image data once ---
    logger.info("\n--- Loading All Image Data ---")
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
    example_images = {label: data[0] if data else None for label, data in all_image_data.items()}

    # --- Loop over each model, metric, and layer index ---
    for model_name in models_to_test:
        logger.info(f"\n{'@' * 20} Processing Model: {model_name} {'@' * 20}")
        cfg.model_name = model_name
        is_lpips_model = "lpips" in model_name

        metrics_for_model = [model_name] if is_lpips_model else metrics_to_test

        for metric_name in metrics_for_model:
            logger.info(f"\n{'#' * 20} Processing Metric: {metric_name.upper()} {'#' * 20}")
            cfg.similarity_metric = metric_name

            for layer_idx in layer_indices_map[model_name]:
                logger.info(f"\n{'=' * 20} Processing Layer Index: {layer_idx} {'=' * 20}")
                cfg.embedding_layer = layer_idx

                all_scores = {}
                all_embeddings = {}

                # --- Get Embeddings (or images for LPIPS) ---
                if is_lpips_model:
                    # For LPIPS, the "embeddings" are the raw images.
                    logger.info("\n--- Using Raw Images for LPIPS model ---")
                    for label, images in all_image_data.items():
                        all_embeddings[label] = np.array(images) if images else np.array([])
                        logger.info(f"Found {len(images)} {label.lower()} images.")
                else:
                    # For other models, compute embeddings.
                    logger.info("\n--- Computing Embeddings ---")
                    embedding_extractor = ImageEmbeddingExtractor(cfg)
                    ref_embeddings = embedding_extractor.extract_from_references()
                    all_embeddings = {
                        "Reference": ref_embeddings,
                        "Manual": embedding_extractor.extract_from_frames(manual_images, len(manual_images)),
                        "Optimized": embedding_extractor.extract_from_frames(optimized_images, len(optimized_images)),
                        "Toy": embedding_extractor.extract_from_frames(toy_images, len(toy_images))
                    }
                    for name, embs in all_embeddings.items():
                        logger.info(f"Found {len(embs) if embs is not None else 0} {name.lower()} embeddings.")

                # --- Calculate Scores ---
                logger.info("\n--- Calculating Scores ---")
                ref_data = all_embeddings["Reference"]
                if ref_data.shape[0] == 0:
                    logger.warning("Reference data is empty. Skipping score calculation for this configuration.")
                    continue

                precomputed_kwargs = {}
                if not is_lpips_model:
                    precomputed_kwargs = precompute_matric_args(cfg, ref_data)

                for label, target_data in all_embeddings.items():
                    if target_data.shape[0] == 0:
                        all_scores[label] = np.array([])
                        continue

                    # Pass all embeddings/images at once. The metric function handles aggregation.
                    # This now includes evaluating the reference data against itself.
                    score = similarity(
                        tuning_cfg=cfg,
                        ref_embeddings=ref_data,
                        synthetic_embeddings=target_data,
                        **precomputed_kwargs
                    )
                    # Since similarity now returns a single aggregated score, we create an array with that score.
                    # This maintains the structure for plotting, though it represents a single value for the whole set.
                    all_scores[label] = np.array([score])

                for name, scores in all_scores.items():
                    if len(scores) > 0:
                        logger.info(f"{name} Images Score: {np.mean(scores):.4f}")

                # --- Plotting ---
                logger.info("\n--- Plotting Results ---")
                model_filename_part = model_name.replace('/', '_')
                layer_name = 'img' if is_lpips_model else ('final' if layer_idx == -1 else str(layer_idx))
                plot_filename = f"metrics_comparison_{model_filename_part}_{metric_name}_layer_{layer_name}.png"
                plot_path = output_dir / plot_filename
                plot_scores_with_images(scores=all_scores, images=example_images, output_path=plot_path,
                                        model_name=cfg.model_name, metric_name=metric_name, layer_name=layer_name)

    logger.info(f"\nAll model, metric, and layer evaluations complete. Plots are saved in '{output_dir}'.")


if __name__ == "__main__":
    main()