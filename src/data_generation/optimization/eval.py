import os

import optuna
import optuna.visualization as vis

from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation.optimization.embeddings import ImageEmbeddingExtractor
from data_generation.video import generate_video
from plotting.plotting import visualize_embeddings


def evaluate_results(tuning_config_path: str, output_dir: str):
    """
    Loads the results of a completed optimization study and generates evaluation artifacts.
    """
    print(f"\n{'=' * 80}\nStarting EVALUATION for: {tuning_config_path}\n{'=' * 80}")

    print("--- Loading configurations and study results ---")
    tuning_cfg = TuningConfig.load(tuning_config_path)

    # Load the best synthetic config found during optimization
    best_cfg = SyntheticDataConfig.load(tuning_cfg.output_config_file)
    print(f"Loaded best configuration from: {tuning_cfg.output_config_file}")

    # Load the completed Optuna study from its database file
    study_db_path = f"sqlite:///{os.path.join(tuning_cfg.temp_dir, f'{tuning_cfg.output_config_id}.db')}"
    try:
        study = optuna.load_study(study_name=tuning_cfg.output_config_id, storage=study_db_path)
        print(f"Loaded Optuna study from: {study_db_path}")
    except KeyError:
        print(f"ERROR: Could not find study '{tuning_cfg.output_config_id}' in the database file.")
        print("Please ensure you have run the optimization script first.")
        return

    eval_config(best_cfg, tuning_cfg, output_dir)

    # Optimization history plot
    try:
        plot_output_dir = os.path.join(output_dir, "plots")
        history_path = os.path.join(plot_output_dir, "optimization_history.html")
        vis.plot_optimization_history(study).write_html(history_path)
        print(f"Optimization history saved to {history_path}")
    except ImportError:
        print("Install 'optuna[visualization]' to enable the progress plot.")
    except Exception as e:
        print(f"Could not generate optimization history plot: {e}")

    print("Evaluation complete.")


def eval_config(cfg, tuning_cfg, output_dir):

    print("\n--- Setting up model for evaluation ---")
    embedding_extractor = ImageEmbeddingExtractor(tuning_cfg)

    print("\n--- Generating final video and embeddings ---")
    frames = generate_video(cfg, output_dir)
    reference_vecs = embedding_extractor.extract_from_references()
    synthetic_vecs = embedding_extractor.extract_from_frames(frames, tuning_cfg.num_compare_frames)

    print("\n--- Creating visualizations ---")
    plot_output_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    visualize_embeddings(cfg, tuning_cfg, reference_vecs, synthetic_vecs, plot_output_dir)
    print(f"Embedding plot saved in {plot_output_dir}")
