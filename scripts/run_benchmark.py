import os
import pandas as pd
from tqdm import tqdm
import logging
import numpy as np

from mt.benchmark.dataset import BenchmarkDataset
from mt.benchmark import metrics
from mt.benchmark.models.factory import setup_model_factory
from mt.plotting.debugging import plot_gt_pred_overlays

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark(dataset_path: str, results_dir: str, models_to_run: list[str]):
    """
    Runs the full benchmark on all models and saves the results.
    """
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = BenchmarkDataset(dataset_path)

    factory = setup_model_factory()
    available_models = factory.get_available_models()
    logger.info(f"Available models: {available_models}")

    # Filter models to run based on user input and availability
    models_to_run_filtered = [m for m in models_to_run if m in available_models]
    if not models_to_run_filtered:
        logger.error("None of the specified models are available. Exiting.")
        return

    logger.info(f"Models to run: {models_to_run_filtered}")
    models = [factory.create_model(name) for name in models_to_run_filtered]

    results = []

    for model in models:
        logger.info(f"--- Benchmarking model: {model.model_name} ---")
        all_seg_metrics = []
        all_downstream_metrics = []

        for idx, (image, gt_mask, _) in enumerate(tqdm(dataset, desc=f"Evaluating {model.model_name}")):
            pred_mask = model.predict(image)
            # pred_mask = gt_mask # For sanity check

            if pred_mask is None or gt_mask is None:
                logger.warning(f"Model {model.model_name} returned None - Skipping image.")
                continue

            seg_metrics = metrics.calculate_segmentation_metrics(pred_mask, gt_mask)
            down_metrics = metrics.calculate_downstream_metrics(pred_mask, gt_mask)

            all_seg_metrics.append(seg_metrics)
            all_downstream_metrics.append(down_metrics)

            save_to_path = os.path.join("plots", "benchmark", f"{model.model_name}")
            os.makedirs(save_to_path, exist_ok=True)
            image_name = os.path.basename(dataset.get_image_path(idx))
            save_path = os.path.join(save_to_path, f"{image_name}_overlay.png")
            # Extract mean IoU and F1@0.5 for overlay
            iou = seg_metrics.get('IoU_mean', None)
            f1 = seg_metrics.get('F1@0.50', None)
            plot_gt_pred_overlays(image, gt_mask, pred_mask, boundary=True, thickness=2, alpha=0.6, save_path=save_path, iou=iou, f1=f1)


        if all_seg_metrics:
            avg_seg_metrics = pd.DataFrame(all_seg_metrics).mean().to_dict()
            avg_downstream_metrics = pd.DataFrame(all_downstream_metrics).mean().to_dict()

            model_results = {
                "Model": model.model_name,
                "IoU_mean": avg_seg_metrics.get('IoU_mean', 0),
                "IoU_median": avg_seg_metrics.get('IoU_median', 0),
                "AP50-95": avg_seg_metrics.get('AP50-95', 0),
                "AP@.50": avg_seg_metrics.get('AP@0.50', 0),
                "AP@.75": avg_seg_metrics.get('AP@0.75', 0),
                "AP@.90": avg_seg_metrics.get('AP@0.90', 0),
                "F1@.50": avg_seg_metrics.get('F1@0.50', 0),
                "F1@.75": avg_seg_metrics.get('F1@0.75', 0),
                "Dice@.50": avg_seg_metrics.get('Dice@0.50', 0),
                "Dice@.75": avg_seg_metrics.get('Dice@0.75', 0),
                "BF1@.50": avg_seg_metrics.get('BF1@0.50', 0),
                "BF1@.75": avg_seg_metrics.get('BF1@0.75', 0),
                "PQ@.50": avg_seg_metrics.get('PQ@0.50', 0),
                "SQ@.50": avg_seg_metrics.get('SQ@0.50', 0),
                "DQ@.50": avg_seg_metrics.get('DQ@0.50', 0),
                "Hausdorff@.50": avg_seg_metrics.get('Hausdorff@0.50', np.nan),
                "Hausdorff@.75": avg_seg_metrics.get('Hausdorff@0.75', np.nan),
                "ASSD@.50": avg_seg_metrics.get('ASSD@0.50', np.nan),
                "ASSD@.75": avg_seg_metrics.get('ASSD@0.75', np.nan),
                "Count AbsErr": avg_seg_metrics.get('CountAbsErr', np.nan),
                "Count RelErr": avg_seg_metrics.get('CountRelErr', np.nan),
                "Length KS": avg_downstream_metrics.get('Length_KS', np.nan),
                "Length KL": avg_downstream_metrics.get('Length_KL', np.nan),
                "Length EMD": avg_downstream_metrics.get('Length_EMD', np.nan),
            }
            results.append(model_results)
        else:
            logger.warning(f"No valid predictions made by {model.model_name} across the dataset. Skipping.")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        output_path = os.path.join(results_dir, "benchmark_results.csv")
        results_df.to_csv(output_path, index=False, float_format='%.2f')
        logger.info(f"Benchmark complete. Results saved to {output_path}")
        print("\nBenchmark Results:")
        print(results_df.to_string(index=False))
    else:
        logger.info("Benchmark finished, but no results were generated.")


if __name__ == "__main__":
    DATASET_PATH = "data/SynMT/synthetic/full"
    RESULTS_DIR = "results"

    # Define which models to run here
    MODELS_TO_RUN = [
        # "AnyStar",
        # "CellSAM",
        "Cellpose-SAM",
        # "DRIFT",
        # "FIESTA",
        # "MicroSAM",
        # "SIFINE",
        # "SOAX",
        # "StarDist",
    ]

    run_benchmark(
        dataset_path=DATASET_PATH,
        results_dir=RESULTS_DIR,
        models_to_run=MODELS_TO_RUN
    )