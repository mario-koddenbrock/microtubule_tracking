# scripts/run_benchmark.py
import os
import pandas as pd
from tqdm import tqdm
import logging
import numpy as np

from benchmark.dataset import BenchmarkDataset
from benchmark.metrics import calculate_segmentation_metrics, calculate_downstream_metrics
from benchmark.models.anystar import AnyStar
from benchmark.models.cellsam import CellSAM
from benchmark.models.cellpose_sam import CellposeSAM
from benchmark.models.drift import DRIFT
from benchmark.models.fiesta import FIESTA
from benchmark.models.musam import MuSAM
from benchmark.models.sifine import SIFINE
from benchmark.models.soax import SOAX
from benchmark.models.stardist import StarDist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark(dataset_path: str, results_dir: str):
    """
    Runs the full benchmark on all models and saves the results.
    """
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = BenchmarkDataset(dataset_path)

    models = [
        FIESTA(), SOAX(), SIFINE(), DRIFT(), CellSAM(),
        AnyStar(), MuSAM(), CellposeSAM(), StarDist()
    ]

    results = []

    for model in models:
        logger.info(f"--- Benchmarking model: {model.model_name} ---")
        all_seg_metrics = []
        all_downstream_metrics = []

        for image, gt_mask, _ in tqdm(dataset, desc=f"Evaluating {model.model_name}"):
            # Predict
            pred_mask_instances = model.predict(image)

            # Convert labeled GT mask to a stack of binary instance masks
            instance_labels = np.unique(gt_mask)
            instance_labels = instance_labels[instance_labels != 0]
            gt_mask_instances = np.array([gt_mask == i for i in instance_labels])

            # Calculate metrics for the current image
            if pred_mask_instances.size > 0 and gt_mask_instances.size > 0:
                seg_metrics = calculate_segmentation_metrics(pred_mask_instances, gt_mask_instances)
                downstream_metrics = calculate_downstream_metrics(pred_mask_instances, gt_mask_instances)
                all_seg_metrics.append(seg_metrics)
                all_downstream_metrics.append(downstream_metrics)

        # Average metrics over the entire dataset
        if all_seg_metrics:
            avg_seg_metrics = pd.DataFrame(all_seg_metrics).mean().to_dict()
            avg_downstream_metrics = pd.DataFrame(all_downstream_metrics).mean().to_dict()

            model_results = {
                "Model": model.model_name,
                "AP": avg_seg_metrics.get('AP', 0),
                "F1@.50": avg_seg_metrics.get('F1@0.50', 0),
                "F1@.75": avg_seg_metrics.get('F1@0.75', 0),
                "Boundary F1@.50": avg_seg_metrics.get('BF1@0.50', 0),
                "Boundary F1@.75": avg_seg_metrics.get('BF1@0.75', 0),
                "Length KS": avg_downstream_metrics.get('Length_KS', np.nan),
                "Length KL": avg_downstream_metrics.get('Length_KL', np.nan),
                # Velocity metrics are removed as requested
            }
            results.append(model_results)
        else:
            logger.warning(f"No valid predictions made by {model.model_name} across the dataset. Skipping.")


    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_path = os.path.join(results_dir, "benchmark_results.csv")
    results_df.to_csv(output_path, index=False, float_format='%.2f')

    logger.info(f"Benchmark complete. Results saved to {output_path}")
    print("\nBenchmark Results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":

    DATASET_PATH = "data/SynMT/synthetic/full"
    RESULTS_DIR = "results"

    run_benchmark(dataset_path=DATASET_PATH, results_dir=RESULTS_DIR)