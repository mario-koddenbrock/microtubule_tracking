import logging
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


def plot_scores(scores: dict, output_path: Path):
    """Generates and saves a bar plot of the average scores."""
    labels = list(scores.keys())
    avg_scores = [np.mean(s) if len(s) > 0 else 0 for s in scores.values()]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, avg_scores, color=['skyblue', 'lightgreen', 'salmon', 'gold'])

    ax.set_ylabel('Average Similarity Score')
    ax.set_title('Average Similarity Score by Data Source vs. Reference')
    ax.set_ylim(0, 1)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Bar plot saved to '{output_path}'")
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
    manual_images = load_images_from_dir(Path("data/synthetic_manual"))
    optimized_images = load_images_from_dir(Path("data/optimization/config_B"))

    logger.info("\n--- Computing Embeddings ---")
    ref_embeddings = embedding_extractor.extract_from_references()
    toy_data: Dict[str, Any] = get_toy_data(embedding_extractor)
    toy_embeddings = toy_data.get("embeddings")
    
    manual_embeddings = embedding_extractor.extract_from_frames(manual_images)
    optimized_embeddings = embedding_extractor.extract_from_frames(optimized_images)
    
    logger.info("\n--- Calculating Scores ---")
    all_scores = {
        "Reference": calculate_similarity_scores(cfg, ref_embeddings, ref_embeddings),
        "Manual": calculate_similarity_scores(cfg, ref_embeddings, manual_embeddings),
        "Optimized": calculate_similarity_scores(cfg, ref_embeddings, optimized_embeddings),
        "Toy": calculate_similarity_scores(cfg, ref_embeddings, toy_embeddings)
    }

    for name, scores in all_scores.items():
        if len(scores) > 0:
            logger.info(f"{name} Images (avg score): {np.mean(scores):.4f}")
        else:
            logger.info(f"{name} Images (avg score): N/A (no data)")

    logger.info("\n--- Plotting Results ---")
    output_dir = Path("plots/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_scores(all_scores, output_dir / "metrics_comparison.png")


if __name__ == "__main__":
    main()