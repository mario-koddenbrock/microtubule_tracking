import io
import logging
from typing import Dict, Optional, Any

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(f"mt.{__name__}")


def get_toy_data(embedding_extractor) -> Dict[str, Optional[Any]]:
    """
    Downloads toy images, extracts their embeddings, and returns them in a structured dictionary.

    Args:
        embedding_extractor: An instance of ImageEmbeddingExtractor.

    Returns:
        A dictionary containing 'images' (List[np.ndarray]), 'labels' (List[str]),
        and 'embeddings' (np.ndarray). Returns None for values if fetching fails.
    """
    logger.info("\n--- Loading toy images for comparison ---")
    toy_image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/1/18/Dog_Breeds.jpg",
        "https://upload.wikimedia.org/wikipedia/en/2/2e/Donald_Duck_-_temper.png",
        "https://www.sleepdr.com/wp-content/uploads/2017/02/pink-noise-can-help-you-sleep.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/bf/Paper_Micrograph_Dark.png",
    ]
    toy_labels = ["dog", "donald duck", "pink noise", "micrograph"]
    toy_images = []
    headers = {
        'User-Agent': 'Microtubule Tracking Reasearch (https://github.com/mario-koddenbrock/microtubule_tracking; mario.koddenbrock@htw-berlin.de)'
    }
    for url in toy_image_urls:
        try:
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            toy_images.append(np.array(img))
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to download toy image from {url}: {e}")
        except Exception as e:
            logger.error(f"An error occurred while processing toy image from {url}: {e}", exc_info=True)

    toy_vecs = None
    if toy_images:
        logger.info(f"Extracting embeddings from {len(toy_images)} toy images.")
        # The extractor expects a list of frames; num_frames=None processes all provided images.
        toy_vecs = embedding_extractor.extract_from_frames(toy_images, num_compare_frames=None)

    return {
        "images": toy_images,
        "labels": toy_labels,
        "embeddings": toy_vecs
    }