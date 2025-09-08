from mt.benchmark.dataset import BenchmarkDataset
from mt.benchmark.models.anystar import AnyStar

# --- Configuration ---
DATASET_DIR = "data/SynMT/synthetic/full"  # Change as needed
OUTPUT_MODEL_DIR = "models/AnyStar/finetuned_anystar"
EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 0.0003

# --- Load data using BenchmarkDataset ---
dataset = BenchmarkDataset(DATASET_DIR)
X = []
Y = []
for i in range(len(dataset)):
    image, mask, _ = dataset[i]
    X.append(image)
    Y.append(mask)

# --- Prepare StarDist3D model for finetuning ---
# Use AnyStar weights as starting point
anystar = AnyStar()
anystar._load_model()  # Ensure model is loaded
base_model = anystar._model  # This is a StarDist3D instance

# --- Training config ---
train_kwargs = dict(
    X=X,
    Y=Y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    save_root=OUTPUT_MODEL_DIR,
    augmenter=None,  # You can add augmenters if needed
)

# --- Finetune ---
print(f"Starting finetuning AnyStar on {len(X)} images...")
base_model.train(**train_kwargs)
print(f"Finetuned model saved to {OUTPUT_MODEL_DIR}")
