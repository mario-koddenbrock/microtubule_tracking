from os import PathLike
from typing import List, Dict

import btrack
import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from tqdm import tqdm

from tracking.base_tracker import BaseTracker

FEATURES = [
  "area",
  "major_axis_length",
  "minor_axis_length",
  "orientation",
  "solidity",
]

TRACKING_UPDATES = [
  "motion",
  # "visual",
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
        super().__init__()
        if btrack is None:
            raise ImportError("'btrack' library is required to use BTrackTracker.")

        self.config_path = config_path
        self.max_search_radius = max_search_radius
        print(f"BTrackTracker initialized with config: {config_path}")


    def _get_centroids(self, mask: np.ndarray) -> Dict[int, np.ndarray]:
        """Calculates the center of mass for each object in a mask."""
        centroids = {}
        properties = regionprops(mask)
        for prop in properties:
            # Return as (row, col) numpy array
            centroids[prop.label] = np.array(prop.centroid)
        return centroids

    def track_frames(self, raw_masks: List[np.ndarray], frames: List[np.ndarray] = None) -> List[np.ndarray]:
        """
        Processes a list of raw masks using the btrack library.
        """
        self.reset()
        if not raw_masks: return []

        mask_3d = np.stack(raw_masks, axis=0)
        objects = btrack.utils.segmentation_to_objects(
            mask_3d,
            properties=tuple(FEATURES),
            num_workers=11,  # TODO
        )

        with btrack.BayesianTracker() as tracker:
            tracker.configure(self.config_path)
            tracker.max_search_radius = self.max_search_radius # TODO: Make this configurable
            tracker.tracking_updates = ["MOTION", "VISUAL"]
            tracker.features = FEATURES

            # append the objects to be tracked
            tracker.append(objects)

            # Set the volume (world box) for tracking
            H, W = raw_masks[0].shape
            tracker.volume = ((0, W), (0, H), (-1e5, 1e5))

            # Run the main tracking algorithm
            print("Running btrack optimization...")
            tracker.track(step_size=len(raw_masks), tracking_updates=TRACKING_UPDATES)
            tracker.optimize()

            # Get the final tracklet objects
            final_tracks = tracker.tracks

            data, properties, graph = tracker.to_napari()
            # import napari
            #
            # viewer = napari.Viewer()
            # viewer.add_labels(mask_3d)
            # viewer.add_tracks(data, properties=properties, graph=graph)

        # --- 3. Translation: Map btrack results back to our mask format ---
        print("Reconstructing masks from btrack tracklets...")
        
        # Pre-calculate all centroids from our original masks for faster lookup
        all_raw_centroids = [self._get_centroids(mask) for mask in raw_masks]
        
        # Create empty masks to be filled
        tracked_masks = [np.zeros_like(mask) for mask in raw_masks]

        for tracklet in tqdm(final_tracks, desc="Mapping tracklets to masks"):
            track_id = tracklet.ID
            for i in range(len(tracklet.t)):
                frame_idx = tracklet.t[i]
                
                # btrack gives (y, x) coordinates for the tracklet's center
                tracklet_centroid = np.array([tracklet.y[i], tracklet.x[i]])
                
                # Find the closest object in our original segmentation for this frame
                raw_centroids_in_frame = all_raw_centroids[frame_idx]
                if not raw_centroids_in_frame: continue
                
                raw_ids, centroids_list = zip(*raw_centroids_in_frame.items())
                distances = cdist([tracklet_centroid], list(centroids_list))[0]
                
                closest_idx = np.argmin(distances)
                
                # If the closest object is within our search radius, it's a match
                if distances[closest_idx] <= self.max_search_radius:
                    matched_raw_id = raw_ids[closest_idx]
                    
                    # Paint our high-quality mask with the final btrack ID
                    instance_mask = (raw_masks[frame_idx] == matched_raw_id)
                    tracked_masks[frame_idx][instance_mask] = track_id
        
        return tracked_masks