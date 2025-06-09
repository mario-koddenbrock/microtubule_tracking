import os
from glob import glob
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import (AutoModel, AutoFeatureExtractor, CLIPModel,
                          CLIPImageProcessor, PreTrainedModel,
                          FeatureExtractionMixin)

from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation.video import generate_frames
from file_io.utils import extract_frames


class ImageEmbeddingExtractor:
    """
    A class to extract and optionally reduce image embeddings using transformer models and PCA.

    This class encapsulates model loading, preprocessing, embedding computation, and
    dimensionality reduction. The PCA model is fitted on a set of reference embeddings
    and can then be applied to any subsequent embeddings.

    Attributes:
        config (TuningConfig): The configuration used to initialize the extractor.
        device (torch.device): The device the model is running on ('cuda', 'mps', or 'cpu').
        model (PreTrainedModel): The loaded transformer model.
        processor (FeatureExtractionMixin): The processor for preparing images for the model.
        pca_model (Optional[PCA]): The PCA model fitted on reference embeddings. None if PCA is not used.
    """

    def __init__(self, tuning_cfg: TuningConfig):
        """
        Initializes the ImageEmbeddingExtractor.

        Args:
            tuning_cfg (TuningConfig): Configuration object containing model name,
                                       cache directory, and optional PCA components.
        """
        self.config = tuning_cfg
        self.device = self._get_best_device()
        print(f"Using device: {self.device}")

        self.model, self.processor = self._load_model_and_processor()
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

        # Initialize PCA model, it will be fitted later
        self.pca_model: Optional[PCA] = None

    def _get_best_device(self) -> torch.device:
        """Selects the best available device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model_and_processor(self) -> Tuple[PreTrainedModel, FeatureExtractionMixin]:
        """Loads the model and processor from Hugging Face based on the config."""
        model_name = self.config.model_name
        cache_dir = self.config.hf_cache_dir

        print(f"Loading model and processor for '{model_name}'...")
        if "clip" in model_name.lower():
            model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
            processor = CLIPImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            processor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)

        return model, processor

    def _compute_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Computes the raw (pre-PCA) embedding for a single RGB image.

        Args:
            image (np.ndarray): An image in RGB format (H, W, C).

        Returns:
            np.ndarray: A 1D numpy array representing the raw image embedding.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            if isinstance(self.model, CLIPModel):
                embedding = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    embedding = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, "pooler_output"):
                    embedding = outputs.pooler_output
                else:
                    raise ValueError("Model output does not contain a usable embedding.")

        return embedding.squeeze().cpu().numpy()

    def extract_from_references(self) -> np.ndarray:
        """
        Loads reference videos, computes embeddings, and fits the PCA model.

        This method MUST be called before any other extraction method if you want
        to use PCA, as it trains the PCA model on these reference embeddings.

        Returns:
            np.ndarray: A 2D array of final (possibly PCA-reduced) embeddings.

        Raises:
            ValueError: If no embeddings could be extracted.
        """
        embeddings = []
        ref_dir = self.config.reference_series_dir
        video_files = glob(os.path.join(ref_dir, "*.avi")) + glob(os.path.join(ref_dir, "*.tif"))
        video_files = video_files[:self.config.num_compare_series]

        for video_idx, video_path in enumerate(video_files):
            frames, _ = extract_frames(video_path)
            frames = frames[:self.config.num_compare_frames]

            print(
                f"Reference {video_idx + 1}/{len(video_files)}: {os.path.basename(video_path)} - {len(frames)} frames")

            for frame in tqdm(frames, desc=f"Processing {os.path.basename(video_path)}"):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                emb = self._compute_embedding(frame_rgb)
                embeddings.append(emb)

        if not embeddings:
            raise ValueError("No embeddings were extracted from reference files.")

        raw_embeddings = np.stack(embeddings)
        print(f"Extracted {raw_embeddings.shape[0]} raw reference embeddings of dimension {raw_embeddings.shape[1]}.")

        # Fit and apply PCA if configured
        if hasattr(self.config, 'pca_components') and self.config.pca_components is not None:
            pca_components = min(self.config.pca_components, min(raw_embeddings.shape))
            print(f"Fitting PCA to reduce dimension to {self.config.pca_components}...")
            self.pca_model = PCA(n_components=pca_components)
            reduced_embeddings = self.pca_model.fit_transform(raw_embeddings)
            print(f"PCA fitted. Explained variance ratio: {sum(self.pca_model.explained_variance_ratio_):.4f}")
            print(f"New embedding dimension: {reduced_embeddings.shape[1]}")
            return reduced_embeddings
        else:
            return raw_embeddings

    def _apply_pca_if_available(self, embeddings: np.ndarray) -> np.ndarray:
        """Applies the fitted PCA model to a new set of embeddings."""
        if self.pca_model:
            # print(f"Applying fitted PCA to new embeddings...")
            return self.pca_model.transform(embeddings)
        else:
            print("PCA model not fitted. Returning raw embeddings.")
            return embeddings

    def extract_from_synthetic_config(self, synthetic_cfg: SyntheticDataConfig) -> np.ndarray:
        """
        Generates frames from a SyntheticDataConfig and computes their final embeddings.
        If a PCA model has been fitted, it will be applied.

        Args:
            synthetic_cfg (SyntheticDataConfig): Config for generating synthetic frames.

        Returns:
            np.ndarray: A 2D array of final (possibly PCA-reduced) embeddings.
        """
        raw_embeddings = []
        frame_generator = generate_frames(synthetic_cfg)

        for frame, *_ in tqdm(frame_generator, total=synthetic_cfg.num_frames,
                                    desc="Generating and processing frames"):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            emb = self._compute_embedding(rgb_frame)
            raw_embeddings.append(emb)

        raw_embeddings = np.stack(raw_embeddings)
        return self._apply_pca_if_available(raw_embeddings)

    @staticmethod
    def flatten_spatial_dims(embeddings: np.ndarray) -> np.ndarray:
        """Utility to ensure embeddings are 2D by flattening spatial dimensions."""
        if embeddings.ndim == 3:
            N, H, W = embeddings.shape
            return embeddings.reshape(N, H * W)
        return embeddings