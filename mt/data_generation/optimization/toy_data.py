import io
import logging
from typing import Dict, Optional, Any
from functools import lru_cache

import cv2
import numpy as np
import requests
from PIL import Image


logger = logging.getLogger(f"mt.{__name__}")


@lru_cache(maxsize=None)
def get_toy_data() -> Dict[str, Optional[Any]]:
    """
    Downloads toy images, standardizes them to a 512x512 square, and returns them.
    Caches the result in memory to avoid repeated downloads.

    Args:
        embedding_extractor (ImageEmbeddingExtractor, optional): Not used in this version
                                                                 but kept for signature consistency.

    Returns:
        Dict[str, Optional[Any]]: A dictionary containing the list of processed toy images
                                  and their corresponding labels.
    """
    logger.debug("\n--- Loading and processing toy images for comparison (no cache) ---")
    toy_image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/1/18/Dog_Breeds.jpg",
        "https://upload.wikimedia.org/wikipedia/en/2/2e/Donald_Duck_-_temper.png",
        "https://www.sleepdr.com/wp-content/uploads/2017/02/pink-noise-can-help-you-sleep.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/bf/Paper_Micrograph_Dark.png",
    ]
    toy_labels = ["dog", "donald duck", "pink noise", "micrograph"]
    toy_images = []
    headers = {
        'User-Agent': 'Microtubule Tracking Reasearch (https://github.com/mario-koddenbrock/mt; mario.koddenbrock@htw-berlin.de)'
    }
    for url in toy_image_urls:
        try:
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            img_np = np.array(img)

            # --- Standardize image shape ---
            h, w, _ = img_np.shape
            min_dim = min(h, w)

            # Center crop to a square
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            cropped_img = img_np[start_y:start_y + min_dim, start_x:start_x + min_dim]

            # Resize to a fixed size (e.g., 512x512)
            resized_img = cv2.resize(cropped_img, (512, 512), interpolation=cv2.INTER_AREA)
            toy_images.append(resized_img)

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to download toy image from {url}: {e}")
        except Exception as e:
            logger.error(f"Error processing toy image from {url}: {e}")

    logger.info(f"Loaded and processed {len(toy_images)} toy images.")
    return {"images": toy_images, "labels": toy_labels}
