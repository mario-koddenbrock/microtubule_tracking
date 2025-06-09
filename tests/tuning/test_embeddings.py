import numpy as np
import pytest
from config.tuning import TuningConfig
from data_generation.embeddings import ImageEmbeddingExtractor


@pytest.fixture
def tiny_model_tuning_config(shared_tmp_path):
    """A fixture that provides a TuningConfig pointing to a tiny, fast model."""
    # Use a mock directory for reference videos
    ref_dir = shared_tmp_path / "references"
    ref_dir.mkdir(exist_ok=True)
    # Create a dummy video file that extract_frames can process
    dummy_video_path = ref_dir / "ref.tif"
    dummy_frame = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    # Mock cv2.imwrite if it's complex, or just use a library that works like tifffile
    import tifffile
    tifffile.imwrite(dummy_video_path, dummy_frame)

    # Use a fast, small model for testing
    return TuningConfig(
        model_name="openai/clip-vit-base-patch16",
        reference_series_dir=str(ref_dir),
        num_compare_frames=1,
        pca_components=3  # Enable PCA for one of the tests
    )


def test_extractor_initialization(tiny_model_tuning_config):
    """Tests that the extractor can be initialized without errors."""
    extractor = ImageEmbeddingExtractor(tiny_model_tuning_config)
    assert extractor.model is not None
    assert extractor.processor is not None



def test_no_pca(tiny_model_tuning_config, mocker):
    """Tests that if PCA is disabled, the dimension is the model's original one."""
    tiny_model_tuning_config.pca_components = None  # Disable PCA
    dummy_frame = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    mocker.patch("file_io.utils.extract_frames", return_value=([dummy_frame], None))

    extractor = ImageEmbeddingExtractor(tiny_model_tuning_config)
    ref_embeddings = extractor.extract_from_references()

    assert extractor.pca_model is None
    assert ref_embeddings.shape[1] > 3  # The original dimension is much larger