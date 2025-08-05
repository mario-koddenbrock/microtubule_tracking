import logging
import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
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


def plot_scores_with_images(scores: Dict[str, np.ndarray], images: Dict[str, np.ndarray], output_path: Path):
    """Generates and saves a box plot of scores with a preprocessed example image and median value for each category."""
    labels = list(scores.keys())
    # Filter out categories with no scores before plotting
    score_data = [s for s in scores.values() if s is not None and len(s) > 0]
    valid_labels = [label for label, s in scores.items() if s is not None and len(s) > 0]

    if not score_data:
        logger.warning("No score data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 9))

    # Create box plot and get the dictionary of artists
    boxplot_dict = ax.boxplot(score_data, labels=valid_labels)
    ax.set_ylabel('Similarity Score')
    ax.set_title('Similarity Score Distribution by Data Source')

    # Adjust layout to make space for images at the bottom
    fig.subplots_adjust(bottom=0.25)

    # Add median score text to each box
    for i, median_line in enumerate(boxplot_dict['medians']):
        median_value = median_line.get_ydata()[0]
        ax.text(i + 1, median_value, f'{median_value:.3f}',
                va='center', ha='center', color='white',
                bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.6))

    # Overlay a preprocessed example image for each category below the x-axis
    for i, label in enumerate(valid_labels):
        if label in images and images[label] is not None:
            img = preprocess_image_for_plot(images[label])
            imagebox = OffsetImage(img, zoom=0.75)

            # Anchor the annotation to the x-axis tick
            ab = AnnotationBbox(imagebox, (i + 1, 0),
                                xybox=(0., -60.),  # Offset in points
                                frameon=False,
                                xycoords=('data', 'axes fraction'),
                                boxcoords="offset points",
                                pad=0)
            ax.add_artist(ab)

    plt.savefig(output_path)
    logger.info(f"Box plot saved to '{output_path}'")
    plt.show()


def main():
    """
    Main script to load data, compute embeddings, calculate similarity scores, and plot results.
    """
    _, _, config_path = parse_optimization_args()

    try:
        cfg = TuningConfig.load(config_path)
        cfg.save(config_path)
    except Exception as e:
        print(f"Error: Failed to load or parse the config file '{config_path}'.")
        print(f"Details: {e}")
        sys.exit(1)

    embedding_extractor = ImageEmbeddingExtractor(cfg)

    logger.info("\n--- Loading Data ---")
    reference_images = load_frames_from_videos(Path(cfg.reference_series_dir))
    manual_images = load_images_from_dir(Path("data/synthetic_manual"))
    optimized_images = load_images_from_dir(Path("data/optimization/config_B"))
    toy_data: Dict[str, Any] = get_toy_data(embedding_extractor)
    toy_images = toy_data.get("images", [])

    logger.info("\n--- Computing Embeddings ---")
    ref_embeddings = embedding_extractor.extract_from_references()
    manual_embeddings = embedding_extractor.extract_from_frames(manual_images, len(manual_images))
    optimized_embeddings = embedding_extractor.extract_from_frames(optimized_images, len(optimized_images))
    toy_embeddings = toy_data.get("embeddings")

    logger.info(f"Found {len(ref_embeddings)} reference embeddings.")
    logger.info(f"Found {len(manual_embeddings)} manual embeddings.")
    logger.info(f"Found {len(optimized_embeddings)} optimized embeddings.")
    logger.info(f"Found {len(toy_embeddings)} toy embeddings.")

    logger.info("\n--- Calculating Scores ---")
    all_scores = {
        "Reference": calculate_similarity_scores(cfg, ref_embeddings, ref_embeddings),
        "Manual": calculate_similarity_scores(cfg, ref_embeddings, manual_embeddings),
        "Optimized": calculate_similarity_scores(cfg, ref_embeddings, optimized_embeddings),
        "Toy": calculate_similarity_scores(cfg, ref_embeddings, toy_embeddings)
    }

    for name, scores in all_scores.items():
        if len(scores) > 0:
            logger.info(f"{name} Images (avg score): {np.mean(scores):.4f}, std: {np.std(scores):.4f}")
        else:
            logger.info(f"{name} Images (avg score): N/A (no data)")

    logger.info("\n--- Plotting Results ---")
    example_images = {
        "Reference": reference_images[0] if reference_images else None,
        "Manual": manual_images[0] if manual_images else None,
        "Optimized": optimized_images[0] if optimized_images else None,
        "Toy": toy_images[0] if toy_images else None
    }

    output_dir = Path("plots/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_scores_with_images(all_scores, example_images, output_dir / "metrics_comparison_boxplot.png")


if __name__ == "__main__":
    main()