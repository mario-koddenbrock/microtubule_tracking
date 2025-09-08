import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from scipy.stats import uniform
import random

from mt.benchmark.dataset import BenchmarkDataset
from mt.benchmark import metrics
from mt.benchmark.models.factory import setup_model_factory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def random_search(model_name, model_params, param_space, dataset, n_iter=20, metric_key="IoU_mean"):
    factory = setup_model_factory()
    results = []
    logger.info(f"Starting random search for {model_name} with params {model_params}")
    for i in range(n_iter):
        params = {k: v.rvs() if hasattr(v, 'rvs') else random.uniform(*v) for k, v in param_space.items()}
        params = {k: float(params[k]) for k in params}  # ensure float
        all_params = {**model_params, **params}
        logger.info(f"[{i+1}/{n_iter}] Trying params: {all_params}")
        try:
            model = factory.create_model(model_name, **all_params)
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            continue
        seg_metrics_list = []
        for idx, (image, gt_mask, _) in enumerate(dataset):
            pred_mask = model.predict(image)
            if pred_mask is None or gt_mask is None:
                continue
            seg_metrics = metrics.calculate_segmentation_metrics(pred_mask, gt_mask)
            seg_metrics_list.append(seg_metrics)
        if seg_metrics_list:
            avg_metric = np.mean([m.get(metric_key, 0) for m in seg_metrics_list])
        else:
            avg_metric = 0
        logger.info(f"Params: {all_params} -> {metric_key}: {avg_metric:.4f}")
        results.append({**all_params, metric_key: avg_metric})
    results_df = pd.DataFrame(results)
    best_idx = results_df[metric_key].idxmax()
    best_row = results_df.loc[best_idx]
    logger.info(f"Best params for {model_name} {model_params}: {best_row.to_dict()}")
    return results_df


def main():
    DATASET_PATH = "data/SynMT/synthetic/full"
    N_SAMPLES = 5  # number of images to use for speed
    N_ITER = 50    # number of random search iterations
    RESULTS_DIR = "results/hyperparam_search"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = BenchmarkDataset(DATASET_PATH, num_samples=N_SAMPLES)

    # Parameter search space
    param_space = {
        "prob_thresh": uniform(0.01, 0.95),   # range: 0.01 to 0.96
        "nms_thresh": uniform(0.01, 0.95),    # range: 0.01 to 0.96
    }

    model_configs = [
        {"name": "AnyStar", "params": {}},
        # {"name": "StarDist", "params": {"pretrained": "2D_versatile_fluo", "model_name": "StarDist_2D_fluo"}},
        # {"name": "StarDist", "params": {"pretrained": "2D_versatile_he", "model_name": "StarDist_2D_he"}},
        # {"name": "StarDist", "params": {"pretrained": "2D_paper_dsb2018", "model_name": "StarDist_dsb2018"}},
    ]

    for config in model_configs:
        model_name = config["name"]
        model_params = config["params"]
        # Create a unique name for the output file
        suffix = model_params.get("model_name", model_name)
        results_df = random_search(model_name, model_params, param_space, dataset, n_iter=N_ITER)
        out_csv = os.path.join(RESULTS_DIR, f"{suffix}_random_search.csv")
        results_df.to_csv(out_csv, index=False, float_format="%.4f")
        logger.info(f"Saved results for {model_name} ({suffix}) to {out_csv}")
        print(f"\nBest for {suffix}:\n", results_df.sort_values("IoU_mean", ascending=False).head(1))

if __name__ == "__main__":
    main()
