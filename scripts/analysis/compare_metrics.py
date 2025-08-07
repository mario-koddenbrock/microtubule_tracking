import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

from config.tuning import TuningConfig
from data_generation.optimization.embeddings import ImageEmbeddingExtractor
from data_generation.optimization.toy_data import get_toy_data
from scripts.utils.cli import parse_optimization_args
from scripts.utils.metric_visualization import load_frames_from_videos, load_images_from_dir, \
    calculate_similarity_scores, plot_scores_with_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    models_to_test = ["facebook/dinov2-large", "openai/clip-vit-base-patch32", "lpips-alex", "lpips-vgg"]
    metrics_to_test = [
        "cosine",
        "fid",
        "kid",
        # "jsd",
        "mahalanobis",
    ]
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

        for layer_idx in layer_indices_map[model_name]:
            logger.info(f"\n{'=' * 20} Processing Layer Index: {layer_idx} {'=' * 20}")
            cfg.embedding_layer = layer_idx

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

            for metric_name in metrics_for_model:
                logger.info(f"\n{'#' * 20} Processing Metric: {metric_name.upper()} {'#' * 20}")
                cfg.similarity_metric = metric_name

                all_scores = {}


                for label, target_data in all_embeddings.items():
                    if target_data.shape[0] == 0:
                        all_scores[label] = np.array([])
                        continue

                    scores = calculate_similarity_scores(
                        cfg=cfg,
                        ref_embeddings=ref_data,
                        target_embeddings=target_data
                    )

                    all_scores[label] = scores


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