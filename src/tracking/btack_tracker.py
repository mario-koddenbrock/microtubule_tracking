import logging
import os
from os import PathLike
from typing import List, Dict, Any, Optional

import btrack
import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from tqdm import tqdm

from .base import BaseTracker

logger = logging.getLogger(f"mt.{__name__}")

FEATURES = [
    "area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "solidity",
]

TRACKING_UPDATES = [
    "motion",
    "visual",  # Ensure "visual" is consistently used if intended
]


class BTrackTracker(BaseTracker):
    """
    A tracker that uses the Bayesian framework of `btrack`.
    """

    def __init__(self, config_path: PathLike, max_search_radius: int = 50):
        """
        Initializes the btrack wrapper.

        Args:
            config_path (str): The path to the btrack JSON configuration file.
            max_search_radius (int): The maximum distance (in pixels) to search
                                     for a matching segmented object when
                                     reconstructing the masks from btrack's output.
        """
        logger.info(
            f"Initializing BTrackTracker with config: '{config_path}' and max_search_radius: {max_search_radius}.")

        # Call the base class constructor first
        super().__init__()

        try:
            import btrack  # Re-import here to catch error more locally
        except ImportError:
            msg = "'btrack' library is required to use BTrackTracker. Please install it ('pip install btrack')."
            logger.critical(msg)
            raise ImportError(msg)

        if not os.path.exists(config_path):
            logger.critical(f"btrack configuration file not found at: {config_path}")
            raise FileNotFoundError(f"btrack configuration file not found at: {config_path}")

        self.config_path = config_path
        self.max_search_radius = max_search_radius
        logger.debug("BTrackTracker initialization complete.")

    def _get_centroids(self, mask: np.ndarray) -> Dict[int, np.ndarray]:
        """Calculates the center of mass for each object in a mask."""
        logger.debug(f"Calculating centroids for mask of shape {mask.shape}.")
        centroids = {}
        if np.max(mask) == 0:
            logger.debug("Mask is empty, no centroids to calculate.")
            return centroids

        properties = regionprops(mask)
        for prop in properties:
            # Return as (row, col) numpy array
            centroids[prop.label] = np.array(prop.centroid)
            logger.debug(f"  Object {prop.label}: Centroid {prop.centroid}.")
        logger.debug(f"Extracted {len(centroids)} centroids from mask.")
        return centroids

    def track_frames(self, raw_masks: List[np.ndarray], frames: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Processes a list of raw masks using the btrack library.

        Args:
            raw_masks (List[np.ndarray]): A list of independently segmented masks
                                       where labels are not consistent over time.
            frames (Optional[List[np.ndarray]]): Optional list of original pixel frames.
                                                Not directly used by btrack but required by BaseTracker interface.

        Returns:
            List[np.ndarray]: A list of re-labeled masks with consistent tracking IDs.
        """
        # Call the base class's track_frames for initial validation and logging
        # This also ensures self.reset() is called by the base.
        try:
            super().track_frames(raw_masks, frames if frames is not None else [])  # Pass empty list if frames is None
        except Exception as e:
            logger.error(f"BaseTracker validation failed: {e}", exc_info=True)
            raise  # Re-raise, as validation failure is critical

        if not raw_masks:
            logger.warning("No raw masks provided. Returning empty list of tracked masks.")
            return []

        logger.info(f"Starting btrack processing for {len(raw_masks)} frames.")

        mask_3d: np.ndarray
        objects: List[Any]  # btrack.object_model.Objects
        try:
            mask_3d = np.stack(raw_masks, axis=0)
            logger.debug(f"Stacked raw masks into 3D array of shape {mask_3d.shape}.")

            # Ensure num_workers is reasonable given the environment (e.g., avoid too many for small tasks)
            # You might want to get this from a config or determine dynamically
            num_workers = os.cpu_count() or 1  # Default to available CPUs
            logger.debug(f"Extracting objects properties using {num_workers} workers with features: {FEATURES}.")
            objects = btrack.utils.segmentation_to_objects(
                mask_3d,
                properties=tuple(FEATURES),
                num_workers=num_workers,
            )
            logger.info(f"Extracted {len(objects)} objects for tracking.")
            if not objects:
                logger.warning("No objects detected from segmentation masks. Returning empty tracked masks.")
                return [np.zeros_like(mask) for mask in raw_masks]  # Return empty masks if no objects
        except Exception as e:
            logger.error(f"Error during object extraction from masks: {e}", exc_info=True)
            raise  # Critical failure

        tracker = None
        try:
            with btrack.BayesianTracker() as tracker:
                tracker.configure(self.config_path)
                tracker.max_search_radius = self.max_search_radius
                tracker.tracking_updates = TRACKING_UPDATES
                tracker.features = FEATURES
                logger.debug(
                    f"btrack configured: max_search_radius={self.max_search_radius}, tracking_updates={TRACKING_UPDATES}, features={FEATURES}.")

                # append the objects to be tracked
                tracker.append(objects)
                logger.debug(f"Appended {len(objects)} objects to btrack for tracking.")

                # Set the volume (world box) for tracking
                H, W = raw_masks[0].shape
                tracker.volume = ((0, W), (0, H), (-1e5, 1e5))  # Z-dim is dummy here
                logger.debug(f"btrack volume set to ((0, {W}), (0, {H}), (-1e5, 1e5)).")

                # Run the main tracking algorithm
                logger.info("Running btrack optimization...")
                # The step_size argument should match the total number of frames you are tracking
                # tracking_updates should also be passed to track if it differs from tracker.tracking_updates
                tracker.track(step_size=len(raw_masks))  # Only pass step_size to track
                tracker.optimize()
                logger.info("btrack optimization complete.")

                # Get the final tracklet objects
                final_tracks = tracker.tracks
                logger.info(f"btrack found {len(final_tracks)} final tracklets.")

                # Optional: napari visualization, commented out
                # data, properties, graph = tracker.to_napari()
                # import napari
                # viewer = napari.Viewer()
                # viewer.add_labels(mask_3d)
                # viewer.add_tracks(data, properties=properties, graph=graph)
        except Exception as e:
            logger.critical(f"Critical error during btrack execution: {e}", exc_info=True)
            # If btrack fails, we cannot produce tracked masks.
            return [np.zeros_like(mask) for mask in raw_masks]  # Return empty masks or re-raise

        # --- 3. Translation: Map btrack results back to our mask format ---
        logger.info("Reconstructing masks from btrack tracklets...")

        # Pre-calculate all centroids from our original masks for faster lookup
        logger.debug("Pre-calculating centroids for all raw masks.")
        all_raw_centroids = [self._get_centroids(mask) for mask in raw_masks]
        logger.debug(f"Pre-computed centroids for {len(all_raw_centroids)} frames.")

        # Create empty masks to be filled with btrack IDs
        tracked_masks = [np.zeros_like(mask, dtype=np.uint16) for mask in raw_masks]  # Ensure uint16 for labels

        no_match_count = 0
        for tracklet in tqdm(final_tracks, desc="Mapping tracklets to masks"):
            track_id = tracklet.ID
            for i in range(len(tracklet.t)):
                frame_idx = tracklet.t[i]

                # Check if frame_idx is within bounds
                if not (0 <= frame_idx < len(raw_masks)):
                    logger.warning(f"Tracklet {track_id} has out-of-bounds frame index {frame_idx}. Skipping.")
                    continue

                # btrack gives (y, x) coordinates for the tracklet's center
                tracklet_centroid = np.array([tracklet.y[i], tracklet.x[i]])

                raw_centroids_in_frame = all_raw_centroids[frame_idx]

                if not raw_centroids_in_frame:
                    logger.debug(f"Frame {frame_idx}: No raw objects to match for tracklet {track_id}.")
                    no_match_count += 1
                    continue

                raw_ids_list: List[int] = list(raw_centroids_in_frame.keys())
                centroids_array: np.ndarray = np.array(list(raw_centroids_in_frame.values()))

                if centroids_array.ndim == 1:  # Handle case of a single object (1, 2)
                    centroids_array = centroids_array.reshape(1, -1)

                distances = cdist([tracklet_centroid], centroids_array)[0]

                closest_idx = np.argmin(distances)
                closest_distance = distances[closest_idx]

                # If the closest object is within our search radius, it's a match
                if closest_distance <= self.max_search_radius:
                    matched_raw_id = raw_ids_list[closest_idx]

                    # Paint our high-quality mask with the final btrack ID
                    instance_mask = (raw_masks[frame_idx] == matched_raw_id)
                    tracked_masks[frame_idx][instance_mask] = track_id
                    logger.debug(
                        f"Tracklet {track_id} (Frame {frame_idx}): Matched raw ID {matched_raw_id} at dist {closest_distance:.2f}.")
                else:
                    logger.debug(
                        f"Tracklet {track_id} (Frame {frame_idx}): No object found within max_search_radius ({self.max_search_radius}) for centroid {tracklet_centroid.tolist()}. Closest at {closest_distance:.2f}.")
                    no_match_count += 1

        if no_match_count > 0:
            logger.warning(
                f"Finished mask reconstruction. {no_match_count} tracklet points could not be matched to a raw object within max_search_radius.")

        logger.info(f"Mask reconstruction complete. Returning {len(tracked_masks)} tracked masks.")
        return tracked_masks