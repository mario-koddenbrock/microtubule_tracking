import os.path
from functools import partial
from pathlib import Path

import optuna

from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation import embeddings as embd
from data_generation.optimization.objective import objective
from data_generation.video import generate_video
from plotting.plotting import visualize_embeddings


def run(tuning_config_path: str):

    tuning_cfg = TuningConfig.load(tuning_config_path)
    tuning_cfg.validate()
    tuning_cfg.to_json(tuning_config_path)  # persist any defaults

    extractor, model = embd.get_model(tuning_cfg)
    ref_embeddings = embd.load_references(tuning_cfg, model, extractor)

    study = optuna.create_study(direction=tuning_cfg.direction, study_name=tuning_cfg.output_config_id)

    objective_fcn = partial(
        objective, tuning_cfg=tuning_cfg, ref_embs=ref_embeddings,
        model=model, extractor=extractor)

    study.optimize(objective_fcn, n_trials=tuning_cfg.num_trials)

    best_cfg = SyntheticDataConfig(**study.best_trial.params)
    best_cfg.id = tuning_cfg.output_config_id
    best_cfg.num_frames = tuning_cfg.output_config_num_frames
    best_cfg.validate()
    best_cfg.to_json(tuning_cfg.output_config_file)
    print(f"✓ Best config saved to {tuning_cfg.output_config_file}")

    # Optional: optimization history plot
    try:
        import optuna.visualization as vis
        vis.plot_optimization_history(study).write_html("../../.temp/optimization_history.html")
        print("✓ Optimisation history saved to optimisation_history.html")
    except ImportError:
        print("Install ‘optuna[visualization]’ to enable the progress plot.")


def evaluation(tuning_config_path: str, output_dir: str = ".temp/"):

    tuning_cfg = TuningConfig.load(tuning_config_path)
    best_cfg = SyntheticDataConfig.load(tuning_cfg.output_config_file)
    video_path, gt_path_json, gt_path_video = generate_video(best_cfg, output_dir)
    print("✓ Best video and ground truth saved to:")
    print(f"  Video: {video_path}")
    print(f"  Ground Truth JSON: {gt_path_json}")
    print(f"  Ground Truth Video: {gt_path_video}")

    # Optional: plot t-SNE projection of embeddings
    extractor, model = embd.get_model(tuning_cfg)
    ref_embeddings = embd.load_references(tuning_cfg, model, extractor)
    ref_vecs = embd.flatten(ref_embeddings)

    # TODO load the best embeddings from the video frames -> faster
    best_vecs = embd.from_cfg(best_cfg, model, extractor)
    output_dir = output_dir.replace("data", "plots")
    output_dir = output_dir.replace("synthetic/", "")
    visualize_embeddings(best_cfg, ref_vecs, best_vecs, output_dir)


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    output_dir = os.path.join(project_root, "data", "synthetic")

    cfg_path_A = os.path.join(project_root, "config", "tuning_config_A.json")
    cfg_path_B = os.path.join(project_root, "config", "tuning_config_B.json")

    run(cfg_path_A)
    evaluation(cfg_path_A, output_dir)

    run(cfg_path_B)
    evaluation(cfg_path_B, output_dir)
