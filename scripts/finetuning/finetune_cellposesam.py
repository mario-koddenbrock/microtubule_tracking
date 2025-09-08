import os
import subprocess
import argparse

from mt.benchmark.dataset import BenchmarkDataset

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Finetune CellposeSAM on a dataset.")
parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory (should contain images/ and image_masks/)')
args = parser.parse_args()

# --- Configuration ---
DATASET_DIR = args.dataset_dir
OUTPUT_MODEL_NAME = "finetuned_cellposesam"
EPOCHS = 50
LEARNING_RATE = 0.2
BATCH_SIZE = 8

# --- Verify dataset with BenchmarkDataset ---
dataset = BenchmarkDataset(DATASET_DIR)
if len(dataset) == 0:
    raise RuntimeError(f"No images found in dataset: {DATASET_DIR}")

# --- Prepare Cellpose training directory ---
TRAIN_DIR = os.path.join(DATASET_DIR, "cellpose_train")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "image_masks")

os.makedirs(TRAIN_DIR, exist_ok=True)

# Copy/symlink images
for fname in os.listdir(IMAGES_DIR):
    src = os.path.join(IMAGES_DIR, fname)
    dst = os.path.join(TRAIN_DIR, fname)
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(src), dst)

# Copy/symlink and rename masks
for fname in os.listdir(MASKS_DIR):
    src = os.path.join(MASKS_DIR, fname)
    base, ext = os.path.splitext(fname)
    dst = os.path.join(TRAIN_DIR, f"{base}_masks{ext}")
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(src), dst)

# --- Call cellpose CLI for training ---
cmd = [
    "cellpose",
    "--train",
    "--dir", TRAIN_DIR,
    "--pretrained_model", "cpsam",
    "--model_name_out", OUTPUT_MODEL_NAME,
    "--n_epochs", str(EPOCHS),
    "--learning_rate", str(LEARNING_RATE),
    "--train_batch_size", str(BATCH_SIZE),
]

print(f"Running CellposeSAM finetuning via CLI:\n{' '.join(cmd)}\n")
ret = subprocess.run(cmd)
if ret.returncode != 0:
    print("CellposeSAM training failed. Please check your dataset and CLI output.")
else:
    print(f"Finetuned model saved as '{OUTPUT_MODEL_NAME}' in the default cellpose models directory (~/.cellpose/models/)")
