import os
import pandas as pd
from tqdm import tqdm
import logging
import numpy as np
import time

from mt.benchmark.dataset import BenchmarkDataset
from mt.benchmark import metrics
from mt.benchmark.models.factory import setup_model_factory
from mt.plotting.debugging import plot_gt_pred_overlays

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark(dataset_path: str, results_dir: str, models_to_run: list):
    """
    Runs the full benchmark on all models and saves the results.
    """
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = BenchmarkDataset(dataset_path, num_samples=10)

    factory = setup_model_factory()
    available_models = factory.get_available_models()
    logger.info(f"Available models: {available_models}")

    # Create model instances from the configuration
    models = []
    for model_config in models_to_run:
        name = model_config["name"]
        params = model_config.get("params", {})
        if name in available_models:
            try:
                models.append(factory.create_model(name, **params))
            except Exception as e:
                logger.error(f"Failed to create model '{name}' with params {params}: {e}")
        else:
            logger.warning(f"Model '{name}' is not available. Skipping.")

    if not models:
        logger.error("No models to run. Exiting.")
        return

    logger.info(f"Models to run: {[m.model_name for m in models]}")

    results = []

    for model in models:
        logger.info(f"--- Benchmarking model: {model.model_name} ---")
        all_seg_metrics = []
        all_downstream_metrics = []
        all_processing_times = []

        for idx, (image, gt_mask, _) in enumerate(
            tqdm(dataset, desc=f"Evaluating {model.model_name}")
        ):
            # Time the prediction
            start_time = time.time()
            pred_mask = model.predict(image)
            end_time = time.time()
            processing_time = end_time - start_time
            all_processing_times.append(processing_time)

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
            iou = seg_metrics.get("IoU_mean", None)
            f1 = seg_metrics.get("F1@0.50", None)
            plot_gt_pred_overlays(
                image,
                gt_mask,
                pred_mask,
                boundary=True,
                thickness=2,
                alpha=0.6,
                save_path=save_path,
                iou=iou,
                f1=f1,
            )

            # Save image + prediction overlay only (no title/metrics, no subplots)
            simple_pred_path = os.path.join(save_to_path, f"{image_name}_predonly.png")
            plot_gt_pred_overlays(
                image,
                None,  # No GT overlay
                pred_mask,
                boundary=True,
                thickness=2,
                alpha=0.8,
                save_path=simple_pred_path,
                instance_seg=True,
            )

        if all_seg_metrics:
            avg_seg_metrics = pd.DataFrame(all_seg_metrics).mean().to_dict()
            avg_downstream_metrics = pd.DataFrame(all_downstream_metrics).mean().to_dict()

            # Calculate speed metrics
            avg_processing_time = np.mean(all_processing_times)
            images_per_second = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

            logger.info(
                f"{model.model_name} - Avg processing time: {avg_processing_time:.3f}s, Images/sec: {images_per_second:.2f}"
            )

            model_results = {
                "Model": model.model_name,
                "Avg Images/sec": images_per_second,
                "IoU_mean": avg_seg_metrics.get("IoU_mean", 0),
                "IoU_median": avg_seg_metrics.get("IoU_median", 0),
                "AP50-95": avg_seg_metrics.get("AP50-95", 0),
                "AP@.50": avg_seg_metrics.get("AP@0.50", 0),
                "F1@.50": avg_seg_metrics.get("F1@0.50", 0),
                "Dice@.50": avg_seg_metrics.get("Dice@0.50", 0),
                "BF1@.50": avg_seg_metrics.get("BF1@0.50", 0),
                "PQ@.50": avg_seg_metrics.get("PQ@0.50", 0),
                "SQ@.50": avg_seg_metrics.get("SQ@0.50", 0),
                "DQ@.50": avg_seg_metrics.get("DQ@0.50", 0),
                "Hausdorff@.50": avg_seg_metrics.get("Hausdorff@0.50", np.nan),
                "ASSD@.50": avg_seg_metrics.get("ASSD@0.50", np.nan),
                "Count AbsErr": avg_seg_metrics.get("CountAbsErr", np.nan),
                "Count RelErr": avg_seg_metrics.get("CountRelErr", np.nan),
                "Length KS": avg_downstream_metrics.get("Length_KS", np.nan),
                "Length KL": avg_downstream_metrics.get("Length_KL", np.nan),
                "Length EMD": avg_downstream_metrics.get("Length_EMD", np.nan),
                "Curvature KS": avg_downstream_metrics.get("Curvature_KS", np.nan),
                "Curvature KL": avg_downstream_metrics.get("Curvature_KL", np.nan),
                "Curvature EMD": avg_downstream_metrics.get("Curvature_EMD", np.nan),
                "Count pred": avg_downstream_metrics.get("Count_pred", np.nan),
                "Count gt": avg_downstream_metrics.get("Count_gt", np.nan),
            }
            results.append(model_results)
        else:
            logger.warning(
                f"No valid predictions made by {model.model_name} across the dataset. Skipping."
            )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        output_path = os.path.join(results_dir, "benchmark_results.csv")
        results_df.to_csv(output_path, index=False, float_format="%.2f")
        logger.info(f"Benchmark complete. Results saved to {output_path}")
        print("\nBenchmark Results:")
        print(results_df.to_string(index=False))
    else:
        logger.info("Benchmark finished, but no results were generated.")


if __name__ == "__main__":
    DATASET_PATH = "data/SynMT/synthetic/full"
    RESULTS_DIR = "results"

    # Define which models to run here.
    MODELS_TO_RUN = [
        # {
        #     "name": "AnyStar",  # default parameters
        #     "prob_thresh": 0.5,
        #     "nms_thresh": 0.3,
        # },
        # {
        #     "name": "AnyStar",
        #     "params": {
        #         "prob_thresh": 0.0001,  # Much lower to detect more microtubules
        #         "nms_thresh": 0.99,  # Higher NMS to allow more instances
        #         "normalize": True,  # Ensure normalization
        #         "norm_percentiles": (0.1, 99.9),  # More aggressive normalization
        #     },  # Still, MTs aren't segmented. Not star-shaped!
        # },
        # {
        #     "name": "AnyStar",
        #     "params": {
        #         "model_dir": "models/AnyStar/finetuned_anystar",
        #         "model_name": "mtStar",
        #         # Mario's?
        #     },
        # },
        # {
        #     "name": "CellposeSAM",
        #     "params": {
        #         "model_dir": "models/CellposeSAM/finetuned_cellposesam",
        #         "model_name": "mtCellposeSAM",
        #     },
        # },
        # {
        #     "name": "StarDist",
        #     "params": {
        #         "pretrained": "2D_versatile_fluo",
        #         "model_name": "StarDist_2D_fluo",
        #         "prob_thresh": 0.061613,
        #         "nms_thresh": 0.8974,
        #     },
        # },
        # {
        #     "name": "StarDist",
        #     "params": {
        #         "pretrained": "2D_versatile_he",
        #         "model_name": "StarDist_2D_he",
        #         "prob_thresh": 0.020277,
        #         "nms_thresh": 0.63537,
        #     },
        # },
        # {
        #     "name": "StarDist",
        #     "params": {
        #         "pretrained": "2D_paper_dsb2018",
        #         "model_name": "StarDist_dsb2018",
        #         "prob_thresh": 0.097102,
        #         "nms_thresh": 0.85354,
        #     },
        # },
        # {
        #     "name": "SAM",
        #     "params": {
        #         "points_per_batch": 64,  # only affects speed
        #         "pred_iou_thresh": 0.88,  # default 0.88
        #         "stability_score_thresh": 0.95,  # default 0.95
        #         "min_mask_region_area": 0,  # default 0
        #     },
        # },
        # {
        #     "name": "SAM2",
        #     "params": {
        #         "points_per_batch": 64,  # only affects speed
        #         "pred_iou_thresh": 0.8,  # default 0.8
        #         "stability_score_thresh": 0.95,  # default 0.95
        #         "min_mask_region_area": 0,  # default 0
        #     },
        # },
        # {"name": "Cellpose-SAM"},
        # {
        #     "name": "MicroSAM",
        #     "params": {
        #         "model_type": "vit_l_lm",
        #         # Default μSAM parameters -> Won't lead to any segmentations on SynMT
        #     },
        # },
        {
            "name": "MicroSAM",
            "params": {
                "model_type": "vit_l_lm",
                "center_distance_threshold": 0.9,
                "boundary_distance_threshold": 0.9,
                "foreground_threshold": 0.1,
                # "foreground_sm oothing": 0.1,
                # "distance_smoothing": 0.5,
            },
        },
        # {"name": "CellSAM"}, # Needs token from deepcell - did not get it until now
        # {"name": "DRIFT"}, # No pretrained model available
        # {"name": "FIESTA"}, # Only MATLAB version available: https://github.com/fiesta-tud/FIESTA/wiki
        # {"name": "SIFNE"}, # Only MATLAB version available
        # {"name": "SOAX"}, # Only C++ and no pretrained model available
    ]

    run_benchmark(dataset_path=DATASET_PATH, results_dir=RESULTS_DIR, models_to_run=MODELS_TO_RUN)
