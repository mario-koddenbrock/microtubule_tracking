from functools import partial

import numpy as np
import optuna
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoFeatureExtractor

from data_generation.config import TuningConfig, SyntheticDataConfig
from data_generation.main import generate_video
from data_generation.utils import load_reference_embeddings, cfg_to_embeddings
from plotting.plotting import visualize_embeddings


def evaluate(cfg_dict: dict,
             ref_embeddings: np.ndarray | list,
             tune_cfg: TuningConfig,
             model,
             extractor) -> float:
    """
    Build a SyntheticDataConfig from Optuna’s `trial` parameters and compare
    *all* generated frames to the reference embedding bank.

    Returns
    -------
    float
        Mean cosine-similarity averaged (1) over reference embeddings *and*
        (2) over the selected number of frames (tune_cfg.num_compare_frames).
    """
    # 1. Build a minimal, in-memory config with exactly the requested #frames
    synth_cfg = SyntheticDataConfig(**cfg_dict,
                                    id=0,
                                    num_frames=tune_cfg.num_compare_frames)
    synth_cfg.validate()

    # 2. Loop through the embeddings and accumulate per-frame scores
    embeddings = cfg_to_embeddings(synth_cfg, model, extractor)
    frame_scores = [np.max(cosine_similarity(emb.reshape(1, -1), ref_embeddings)[0]) for emb in embeddings]

    # 3. Average across frames
    return float(np.mean(frame_scores))



def objective(trial, tuning_cfg, ref_embs, model, extractor):
    trial_cfg = {
        "grow_amp"   : trial.suggest_float("grow_amp",   *tuning_cfg.grow_amp_range),
        "grow_freq"  : trial.suggest_float("grow_freq",  *tuning_cfg.grow_freq_range),
        "shrink_amp" : trial.suggest_float("shrink_amp", *tuning_cfg.shrink_amp_range),
        "shrink_freq": trial.suggest_float("shrink_freq",*tuning_cfg.shrink_freq_range),
        "motion"     : trial.suggest_float("motion",     *tuning_cfg.motion_range),
        "max_length" : trial.suggest_float("max_length", *tuning_cfg.max_length_range),
        "min_length" : trial.suggest_float("min_length", *tuning_cfg.min_length_range),
        "snr"        : trial.suggest_float("snr",        *tuning_cfg.snr_range),
        "sigma_x"    : trial.suggest_int("sigma_x",      *tuning_cfg.sigma_range),
        "sigma_y"    : trial.suggest_int("sigma_y",      *tuning_cfg.sigma_range),
        "num_tubulus": int(trial.suggest_int("num_tubulus", *tuning_cfg.num_tubulus_range)),
    }
    return evaluate(trial_cfg, ref_embs, tuning_cfg, model, extractor)


def main():

    config_path = "../../config/tuning_config.json"
    tuning_cfg  = TuningConfig.load(config_path)
    tuning_cfg.validate()
    tuning_cfg.to_json(config_path) # persist any defaults

    model = AutoModel.from_pretrained(tuning_cfg.model_name, cache_dir=tuning_cfg.hf_cache_dir)
    extractor = AutoFeatureExtractor.from_pretrained(tuning_cfg.model_name, cache_dir=tuning_cfg.hf_cache_dir)
    ref_embeddings = load_reference_embeddings(tuning_cfg, model, extractor)

    study = optuna.create_study(direction=tuning_cfg.direction)

    objective_fcn = partial(
        objective, tuning_cfg=tuning_cfg, ref_embs=ref_embeddings,
        model=model, extractor=extractor)

    study.optimize(objective_fcn, n_trials=tuning_cfg.num_trials)

    best_cfg = SyntheticDataConfig(**study.best_trial.params)
    best_cfg_path = "../../config/best_synthetic_config.json"
    best_cfg.to_json(best_cfg_path)
    print(f"✓ Best config saved to {best_cfg_path}")

    output_dir = "../../data/synthetic"
    video_path, gt_path, gt_video_path = generate_video(best_cfg, output_dir)
    print("✓ Best video and ground truth saved to:")
    print(f"  Video: {video_path}")
    print(f"  Ground Truth JSON: {gt_path}")
    print(f"  Ground Truth Video: {gt_video_path}")


    # Optional: optimisation history plot
    try:
        import optuna.visualization as vis
        vis.plot_optimization_history(study).write_html("../../.temp/optimization_history.html")
        print("✓ Optimisation history saved to optimisation_history.html")
    except ImportError:
        print("Install ‘optuna[visualization]’ to enable the progress plot.")

    # Optional: plot t-SNE projection of embeddings
    visualize_embeddings(best_cfg, model, extractor, ref_embeddings)

if __name__ == "__main__":
    main()
