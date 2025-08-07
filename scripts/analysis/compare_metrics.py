import logging
import sys
from pathlib import Path

from config.tuning import TuningConfig
from scripts.utils.cli import parse_optimization_args
from scripts.utils.metric_visualization import load_all_data, process_model_evaluation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def main():
    """
    Main script to load data, compute embeddings, calculate similarity, and plot results.
    """
    _, _, config_path = parse_optimization_args()
    try:
        cfg = TuningConfig.load(config_path)
    except Exception as e:
        logger.error(f"Failed to load or parse config file '{config_path}': {e}")
        sys.exit(1)


    models_to_test = [
        # "facebook/dinov2-large",
        # "openai/clip-vit-base-patch32",
        "lpips-alex",
        "lpips-vgg",
    ]
    metrics_to_test = [
        "cosine",
        # "fid",
        # "kid",
        # "mahalanobis",
    ]
    layer_indices_map = {
        "openai/clip-vit-base-patch32": list(range(1, 13)),
        "facebook/dinov2-large": list(range(1, 25)),
        "lpips-alex": [0],  # Dummy layer for image-based metric
        "lpips-vgg": [0],   # Dummy layer for image-based metric
    }

    output_dir = Path("plots/metric_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data and run evaluations ---
    all_image_data = load_all_data(cfg)
    example_images = {label: data[0] for label, data in all_image_data.items() if data}

    for model_name in models_to_test:
        process_model_evaluation(
            cfg=cfg,
            model_name=model_name,
            layer_indices=layer_indices_map[model_name],
            metrics_to_test=metrics_to_test,
            all_image_data=all_image_data,
            example_images=example_images,
            output_dir=output_dir
        )

    logger.info(f"\nEvaluation complete. Plots are saved in '{output_dir}'.")


if __name__ == "__main__":
    main()