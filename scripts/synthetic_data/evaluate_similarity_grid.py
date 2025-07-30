import argparse
import os
from glob import glob
from typing import List, Tuple, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config.tuning import TuningConfig
from data_generation.optimization.embeddings import ImageEmbeddingExtractor
from data_generation.optimization.metrics import similarity
from file_io.utils import extract_frames



def load_all_frames(video_folder: str, max_frames_per_video: int) -> List[Tuple[str, np.ndarray]]:
    """Loads up to a max number of frames from each video in a folder."""
    # video_paths = glob(os.path.join(video_folder, "*.avi")) + glob(os.path.join(video_folder, "*.tif"))
    video_paths = glob(os.path.join(video_folder, "*_video_preview.mp4"))
    if not video_paths:
        print(f"Error: No .avi or .tif videos found in '{video_folder}'")
        exit(1)

    all_frames_data = []
    for video_path in sorted(video_paths):
        try:
            frames, _ = extract_frames(video_path)
            if not frames:
                print(f"Warning: Could not extract frames from {os.path.basename(video_path)}")
                continue

            selected_frames = frames[:max_frames_per_video]
            for frame in selected_frames:
                all_frames_data.append((os.path.basename(video_path), frame))

        except Exception as e:
            print(f"Warning: Failed to process {os.path.basename(video_path)}: {e}")

    return all_frames_data


def precompute_reference_values(tuning_cfg: TuningConfig, all_ref_embeddings: np.ndarray) -> dict:
    """Pre-computes values needed by the similarity metric."""
    metric = getattr(tuning_cfg, 'similarity_metric', 'cosine')
    print(f"Pre-computing reference values for metric: '{metric}'...")

    precomputed_args = {}
    if metric == 'mahalanobis':
        precomputed_args['ref_mean'] = np.mean(all_ref_embeddings, axis=0)
        precomputed_args['ref_inv_cov'] = np.linalg.pinv(np.cov(all_ref_embeddings, rowvar=False))
        print("  - Computed mean and inverse covariance matrix.")
    elif metric in ['ndb', 'jsd']:
        num_bins = getattr(tuning_cfg, 'num_hist_bins', 10)
        hist, bins = np.histogramdd(all_ref_embeddings, bins=num_bins)
        precomputed_args['ref_hist_bins'] = bins
        print(f"  - Computed histogram bins (n={num_bins}).")
        if metric == 'ndb':
            precomputed_args['ref_hist'] = hist
            print("  - Computed reference histogram for NDB.")
        if metric == 'jsd':
            prob = (hist / hist.sum()).flatten()
            prob[prob == 0] = 1e-10
            precomputed_args['ref_prob'] = prob
            print("  - Computed reference probability distribution for JSD.")

    return precomputed_args


def visualize_grid(video_data: Dict, sorted_video_names: List[str], num_cols: int, output_path: str = None):
    """Creates, displays, and optionally saves a grid of frames with their similarity scores."""
    num_videos = len(sorted_video_names)
    if num_videos == 0:
        print("Warning: No data to visualize.")
        return

    num_rows = num_videos

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3.5), squeeze=False)
    fig.suptitle("Frame Similarity Score Grid (Sorted by Average Score)", fontsize=16)

    for i, video_name in enumerate(sorted_video_names):
        frames_and_scores = video_data[video_name]

        avg_score = np.mean([score for _, score in frames_and_scores])
        row_title = f"{video_name}\nAvg Score: {avg_score:.4f}"
        axes[i, 0].set_ylabel(row_title, rotation=0, size='large', labelpad=60, ha='right', va='center')

        for j in range(num_cols):
            ax = axes[i, j]
            if j < len(frames_and_scores):
                frame, score = frames_and_scores[j]
                ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Frame {j}\nScore: {score:.4f}", fontsize=10)
            else:
                ax.axis('off')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0.05, 0, 1, 0.96])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Grid visualization saved to: {output_path}")
    else:
        plt.show()


def run_grid_evaluation(config_path: str, video_folder: str, output_path: str = None):
    """
    Main execution function for the grid evaluation.
    """
    print("=" * 80)
    print("Starting Similarity Grid Evaluation")
    print(f"  - Config File: '{config_path}'")
    print(f"  - Video Folder: '{video_folder}'")
    if output_path:
        print(f"  - Output Path: '{output_path}'")
    print("=" * 80)

    # 1. Load configuration
    try:
        tuning_cfg = TuningConfig.load(config_path)
        tuning_cfg.num_compare_frames = 5
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'")
        exit(1)

    frames_to_analyze = tuning_cfg.num_compare_frames
    print(f"Analyzing the first {frames_to_analyze} frames of each video.")

    # 2. Initialize Model and Extractor
    embedding_extractor = ImageEmbeddingExtractor(tuning_cfg)

    # 3. Establish the GLOBAL reference distribution
    print("\n--- Step 1: Creating global reference distribution ---")
    all_frames_for_ref = load_all_frames(video_folder, max_frames_per_video=100)

    if not all_frames_for_ref:
        print("Error: No valid frames loaded to build reference set. Exiting.")
        exit(1)

    print(f"  - Extracted {len(all_frames_for_ref)} total frames for reference set.")
    all_ref_embeddings = embedding_extractor.extract_from_references()
    precomputed_args = precompute_reference_values(tuning_cfg, all_ref_embeddings)

    # 4. Calculate score for each individual frame
    print("\n--- Step 2: Calculating score for each target frame ---")
    # video_paths = sorted(glob(os.path.join(video_folder, "*.avi")) + glob(os.path.join(video_folder, "*.tif")))
    video_paths = sorted(glob(os.path.join(video_folder, "*_video_preview.mp4")))
    grid_data = {}
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        print(f"  - Processing '{video_name}'...")

        try:
            frames, _ = extract_frames(video_path)
            target_frames = frames[:frames_to_analyze]
            if not target_frames: continue

            grid_data[video_name] = []
            for frame in target_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                emb_raw = embedding_extractor._compute_embedding(frame_rgb)

                if embedding_extractor.pca_model:
                    emb_final = embedding_extractor.pca_model.transform(emb_raw.reshape(1, -1))
                else:
                    emb_final = emb_raw.reshape(1, -1)

                score = similarity(
                    tuning_cfg=tuning_cfg,
                    ref_embeddings=all_ref_embeddings,
                    synthetic_embeddings=emb_final,
                    **precomputed_args
                )
                grid_data[video_name].append((frame, score))
        except Exception as e:
            print(f"Warning: Failed during scoring of {video_name}: {e}")

    # 5. Sort the videos based on their average score
    print("\n--- Step 3: Sorting videos by average score ---")
    video_avg_scores = {
        name: np.mean([s for _, s in scores]) for name, scores in grid_data.items() if scores
    }
    sorted_video_names = sorted(video_avg_scores, key=video_avg_scores.get, reverse=True)

    for name in sorted_video_names:
        print(f"  - Avg Score for '{name}': {video_avg_scores[name]:.4f}")

    # 6. Visualize the results
    print("\n--- Step 4: Generating visualization ---")
    visualize_grid(grid_data, sorted_video_names, num_cols=frames_to_analyze, output_path=output_path)

    print("\nEvaluation complete.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze reference videos, calculate frame-by-frame similarity scores, and visualize them in a sorted grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the tuning configuration JSON or YAML file."
    )
    parser.add_argument(
        "video_folder",
        type=str,
        help="Path to the folder containing reference videos to analyze."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional. Path to save the output grid image (e.g., 'output/grid.png'). If not provided, the plot will be displayed on screen."
    )

    args = parser.parse_args()

    run_grid_evaluation(
        config_path=args.config_path,
        video_folder=args.video_folder,
        output_path=args.output
    )