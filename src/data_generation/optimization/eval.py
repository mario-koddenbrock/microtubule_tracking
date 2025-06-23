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

    # =========================================================================
    # 1. LOAD CONFIGS AND STUDY RESULTS
    # =========================================================================
    print("--- Step 1: Loading configurations and study results ---")
    tuning_cfg = TuningConfig.load(tuning_config_path)

    # Load the best synthetic config found during optimization
    best_cfg = SyntheticDataConfig.load(tuning_cfg.output_config_file)
    print(f"✓ Loaded best configuration from: {tuning_cfg.output_config_file}")

    # Load the completed Optuna study from its database file
    study_db_path = f"sqlite:///{os.path.join(tuning_cfg.temp_dir, f'{tuning_cfg.output_config_id}.db')}"
    try:
        study = optuna.load_study(
            study_name=tuning_cfg.output_config_id,
            storage=study_db_path
        )
        print(f"✓ Loaded Optuna study from: {study_db_path}")
    except KeyError:
        print(f"ERROR: Could not find study '{tuning_cfg.output_config_id}' in the database file.")
        print("Please ensure you have run the optimization script first.")
        return

    # =========================================================================
    # 2. SETUP FOR EMBEDDING GENERATION
    # =========================================================================
    print("\n--- Step 2: Setting up model for evaluation ---")
    # We still need the extractor to generate embeddings for the best config
    # and the reference set for comparison.
    embedding_extractor = ImageEmbeddingExtractor(tuning_cfg)

    # =========================================================================
    # 3. GENERATE VIDEO AND EMBEDDINGS
    # =========================================================================
    print("\n--- Step 3: Generating final video and embeddings ---")
    video_path, gt_path_json, gt_path_video = generate_video(best_cfg, output_dir)
    print("✓ Best video and ground truth saved to:")
    print(f"  Video: {video_path}")

    # Generate embeddings for the final, best video
    best_vecs = embedding_extractor.extract_from_synthetic_config(best_cfg)

    # Re-calculate the reference embeddings for the t-SNE plot comparison
    # This ensures the PCA model is consistent if it's used.
    ref_vecs = embedding_extractor.extract_from_references()

    print(f"✓ Generated embeddings for comparison (Reference: {ref_vecs.shape}, Best: {best_vecs.shape})")

    # =========================================================================
    # 4. CREATE VISUALIZATIONS
    # =========================================================================
    print("\n--- Step 4: Creating visualizations ---")
    plot_output_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    # t-SNE plot
    visualize_embeddings(best_cfg, ref_vecs, best_vecs, plot_output_dir)
    print(f"✓ t-SNE plot saved in {plot_output_dir}")

    # Optimization history plot
    try:
        history_path = os.path.join(plot_output_dir, "optimization_history.html")
        vis.plot_optimization_history(study).write_html(history_path)
        print(f"✓ Optimization history saved to {history_path}")
    except ImportError:
        print("Install 'optuna[visualization]' to enable the progress plot.")
    except Exception as e:
        print(f"Could not generate optimization history plot: {e}")

    print("Evaluation complete.")