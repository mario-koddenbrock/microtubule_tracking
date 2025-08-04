import logging
import os
from glob import glob
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from cellpose import transforms
from cellpose.models import CellposeModel, normalize_default
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import (AutoModel, CLIPModel,
                          CLIPImageProcessor, PreTrainedModel,
                          FeatureExtractionMixin, AutoImageProcessor)

from config.synthetic_data import SyntheticDataConfig
from config.tuning import TuningConfig
from data_generation.video import generate_frames
from file_io.utils import extract_frames

logger = logging.getLogger(f"mt.{__name__}")


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
        logger.debug("Initializing ImageEmbeddingExtractor...")
        self.config = tuning_cfg

        self.device = self._get_best_device()
        logger.debug(f"Using device: {self.device}")

        try:
            self.model, self.processor = self._load_model_and_processor()

            logger.info(f"Model '{self.config.model_name}' loaded and set to evaluation mode on {self.device}.")
        except Exception as e:
            logger.critical(f"Failed to load or initialize model '{self.config.model_name}': {e}", exc_info=True)
            raise  # Re-raise to indicate a critical setup failure

        # Initialize PCA model, it will be fitted later
        self.pca_model: Optional[PCA] = None
        logger.debug("ImageEmbeddingExtractor initialized successfully.")

    def _get_best_device(self) -> torch.device:
        """Selects the best available device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            logger.debug("CUDA is available. Using GPU.")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            logger.debug("MPS (Apple Silicon) is available. Using MPS.")
            return torch.device("mps")
        logger.debug("No GPU (CUDA/MPS) found. Falling back to CPU.")
        return torch.device("cpu")

    def _load_model_and_processor(self) -> Tuple[PreTrainedModel, FeatureExtractionMixin]:
        """Loads the model and processor from Hugging Face based on the config."""
        model_name = self.config.model_name
        cache_dir = self.config.hf_cache_dir

        logger.debug(f"Loading model and processor for '{model_name}' from Hugging Face...")
        try:

            if "clip" in model_name.lower():
                model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
                processor = CLIPImageProcessor.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
                logger.debug(f"Loaded CLIP model and processor: {model_name}.")

            elif "cellpose" in model_name.lower():
                model = CellposeModel(model_type='cellpose_sam', gpu=torch.cuda.is_available(), device=self.device)
                logger.debug(f"Loaded Cellpose model: {model_name}.")

                encoder = model.net.encoder
                encoder.eval()

                return model, encoder
            else:
                model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
                processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
                logger.debug(f"Loaded AutoModel and AutoImageProcessor: {model_name}.")

            model.to(self.device)
            model.eval()  # Set model to evaluation mode

            return model, processor
        except Exception as e:
            logger.error(f"Failed to load model or processor '{model_name}': {e}", exc_info=True)
            raise

    def _compute_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Computes the raw (pre-PCA) embedding for a single RGB image.

        Args:
            image (np.ndarray): An image in RGB format (H, W, C).

        Returns:
            np.ndarray: A 1D numpy array representing the raw image embedding.
        """
        # This is called frequently, so keep logging minimal unless debugging specific issues.
        logger.debug(f"Computing embedding for image of shape {image.shape} (RGB).")

        try:

            if "cellpose" in self.config.model_name.lower():

                x = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
                x = transforms.convert_image(x, channel_axis=None, z_axis=None)
                x = x[np.newaxis, ...]

                normalize_params = normalize_default
                normalize_params["normalize"] = True
                normalize_params["invert"] = False

                x = transforms.normalize_img(x, **normalize_params)
                X = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(self.model.device, dtype=self.model.net.dtype)

                with torch.no_grad():
                    out = self.processor(X)

                return out.detach().squeeze().cpu().to(torch.float32).numpy().flatten()

            else:

                inputs = self.processor(images=image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    if isinstance(self.model, CLIPModel):
                        embedding = self.model.get_image_features(**inputs)
                        logger.debug("Using CLIP model's image features.")
                    else:
                        outputs = self.model(**inputs)
                        if hasattr(outputs, "last_hidden_state"):
                            embedding = outputs.last_hidden_state.mean(dim=1)
                            logger.debug("Using last_hidden_state mean for embedding.")
                        elif hasattr(outputs, "pooler_output"):
                            embedding = outputs.pooler_output
                            logger.debug("Using pooler_output for embedding.")
                        else:
                            msg = "Model output does not contain a usable embedding (last_hidden_state or pooler_output)."
                            logger.error(msg)
                            raise ValueError(msg)

                return embedding.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(f"Error computing embedding for image of shape {image.shape}: {e}", exc_info=True)
            raise

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
        logger.debug("Starting reference embedding extraction and PCA fitting...")
        embeddings = []
        ref_dir = self.config.reference_series_dir
        logger.debug(f"Scanning for reference videos in: {ref_dir}")

        video_files = (glob(os.path.join(ref_dir, "*.avi")) +
                       glob(os.path.join(ref_dir, "*.tif")) +
                          glob(os.path.join(ref_dir, "*.mp4")))

        logger.debug(f"Found {len(video_files)} video files matching patterns.")

        if not video_files:
            msg = f"No reference video files found in directory: {ref_dir}"
            logger.error(msg)
            raise ValueError(msg)

        # Limit to num_compare_series
        original_num_videos = len(video_files)
        video_files = video_files[:self.config.num_compare_series]
        if len(video_files) < original_num_videos:
            logger.warning(f"Limiting reference videos to {len(video_files)} as per config.num_compare_series.")

        for video_idx, video_path in enumerate(video_files):
            logger.info(
                f"Processing reference video {video_idx + 1}/{len(video_files)}: {os.path.basename(video_path)}")
            try:
                frames, _ = extract_frames(video_path)
                original_num_frames = len(frames)
                frames = frames[:self.config.num_compare_frames]
                if len(frames) < original_num_frames:
                    logger.debug(
                        f"Limiting frames from '{os.path.basename(video_path)}' to {len(frames)} as per config.num_compare_frames.")

                if not frames:
                    logger.warning(
                        f"No frames extracted from {os.path.basename(video_path)} or num_compare_frames is 0. Skipping.")
                    continue

                for frame_idx, frame in enumerate(tqdm(frames, desc=f"Processing frames from {os.path.basename(video_path)}")):
                    frame_rgb = self.convert_frame(frame)
                    emb = self._compute_embedding(frame_rgb)
                    embeddings.append(emb)
                    logger.debug(
                        f"Processed frame {frame_idx + 1} from {os.path.basename(video_path)}. Embedding shape: {emb.shape}")
            except Exception as e:
                logger.error(f"Error processing reference video {os.path.basename(video_path)}: {e}", exc_info=True)
                continue  # Try to continue with other videos

        if not embeddings:
            msg = "No embeddings were extracted from any reference files. Cannot proceed."
            logger.critical(msg)
            raise ValueError(msg)

        raw_embeddings = np.stack(embeddings)
        logger.info(
            f"Extracted {raw_embeddings.shape[0]} raw reference embeddings with dimension {raw_embeddings.shape[1]}.")

        # Fit and apply PCA if configured
        if self.config.pca_components is not None:
            # Ensure PCA components are not more than the number of samples or features
            pca_components = min(self.config.pca_components, raw_embeddings.shape[0], raw_embeddings.shape[1])
            if pca_components <= 0:
                logger.warning(
                    f"PCA components effectively zero or negative ({pca_components}). Skipping PCA reduction.")
                self.pca_model = None  # Ensure it's explicitly None
                return raw_embeddings

            logger.info(f"Fitting PCA to reduce dimension to {pca_components} components...")
            self.pca_model = PCA(n_components=pca_components)
            reduced_embeddings = self.pca_model.fit_transform(raw_embeddings)
            explained_variance_ratio = sum(self.pca_model.explained_variance_ratio_)
            logger.info(f"PCA fitted. Explained variance ratio: {explained_variance_ratio:.4f}")
            logger.info(
                f"Reference embeddings reduced from {raw_embeddings.shape[1]} to {reduced_embeddings.shape[1]} dimensions.")
            return reduced_embeddings
        else:
            logger.info("PCA not configured (pca_components is None). Returning raw reference embeddings.")
            return raw_embeddings

    def _apply_pca_if_available(self, embeddings: np.ndarray) -> np.ndarray:
        """Applies the fitted PCA model to a new set of embeddings."""
        if self.pca_model:
            logger.debug(f"Applying fitted PCA to new embeddings (input shape: {embeddings.shape}).")
            try:
                transformed_embeddings = self.pca_model.transform(embeddings)
                logger.debug(f"Embeddings transformed by PCA. New shape: {transformed_embeddings.shape}.")
                return transformed_embeddings
            except Exception as e:
                logger.error(f"Error applying PCA transformation: {e}", exc_info=True)
                # Depending on severity, you might return raw embeddings or re-raise
                raise
        else:
            logger.debug("PCA model not fitted or configured. Returning raw embeddings.")
            return embeddings

    def extract_from_synthetic_config(self, synthetic_cfg: SyntheticDataConfig,
                                      num_compare_frames: int = 1) -> np.ndarray:
        logger.info(
            f"Extracting embeddings from synthetic config (ID: {synthetic_cfg.id}). Comparing {num_compare_frames} frames.")
        raw_embeddings = []
        frame_generator = generate_frames(synthetic_cfg, num_compare_frames)

        for frame, *_ in tqdm(frame_generator, total=num_compare_frames, desc=f"Generating & processing frames for {synthetic_cfg.id}"):

            # frame is already in RGB format, no need to convert
            # rgb_frame = self.convert_frame(frame)
            emb = self._compute_embedding(frame)
            raw_embeddings.append(emb)
            logger.debug(f"Generated and processed frame for synthetic config {synthetic_cfg.id}. Embedding shape: {emb.shape}")


        if not raw_embeddings:
            logger.warning(f"No embeddings generated for synthetic config ID {synthetic_cfg.id}.")
            return np.array([])  # Return empty array if no embeddings could be generated

        raw_embeddings = np.stack(raw_embeddings)
        logger.info(f"Extracted {raw_embeddings.shape[0]} raw embeddings from synthetic config {synthetic_cfg.id}.")
        return self._apply_pca_if_available(raw_embeddings)

    def extract_from_frames(self, frames: List[np.ndarray], num_compare_frames: int = 1) -> np.ndarray:
        logger.debug(
            f"Extracting embeddings from a provided list of {len(frames)} frames. Comparing {num_compare_frames} frames.")
        raw_embeddings = []

        # Limit frames to num_compare_frames if necessary
        frames_to_process = frames[:num_compare_frames]
        if len(frames_to_process) < len(frames):
            logger.debug(f"Limiting processing to first {len(frames_to_process)} frames from the list.")

        try:
            for frame_idx, frame in enumerate(tqdm(frames_to_process, desc="Generating embeddings from frames")):
                rgb_frame = self.convert_frame(frame)
                emb = self._compute_embedding(rgb_frame)
                raw_embeddings.append(emb)
                logger.debug(f"Processed frame {frame_idx + 1} from list. Embedding shape: {emb.shape}")
        except Exception as e:
            logger.error(f"Error during embedding extraction from provided frame list: {e}", exc_info=True)
            raise

        if not raw_embeddings:
            logger.warning("No embeddings generated from the provided frame list.")
            return np.array([])

        raw_embeddings = np.stack(raw_embeddings)
        logger.info(f"Extracted {raw_embeddings.shape[0]} raw embeddings from provided frames.")
        return self._apply_pca_if_available(raw_embeddings)

    def convert_frame(self, frame: np.ndarray) -> np.ndarray:
        """Converts a frame to RGB format if necessary."""
        if frame.ndim == 2:
            logger.debug("Converting grayscale frame to RGB.")
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            # Assuming BGR input for cv2 operations, convert to RGB for models
            # Many image feature extractors expect RGB, while OpenCV defaults to BGR
            # This is a common point of error, so being explicit.
            if frame.dtype == np.uint8:  # Only convert if it's an 8-bit image for typical BGR -> RGB
                logger.debug("Converting BGR frame to RGB.")
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                logger.debug("Frame is already 3-channel. Assuming it's RGB or compatible. Skipping color conversion.")
                return frame
        else:
            logger.warning(f"Unexpected frame dimension/channels: {frame.shape}. Returning as is, may cause issues.")
            return frame

    @staticmethod
    def flatten_spatial_dims(embeddings: np.ndarray) -> np.ndarray:
        """Utility to ensure embeddings are 2D by flattening spatial dimensions."""
        if embeddings.ndim == 3:
            N, H, W = embeddings.shape
            logger.debug(f"Flattening spatial dimensions from {embeddings.shape} to {(N, H * W)}.")
            return embeddings.reshape(N, H * W)
        logger.debug(f"Embeddings are already 2D ({embeddings.shape}). No flattening needed.")
        return embeddings