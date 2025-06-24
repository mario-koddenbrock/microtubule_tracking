from dataclasses import dataclass
from typing import List

import numpy as np

from config.synthetic_data import SyntheticDataConfig
from data_generation.utils import draw_gaussian_line

import matplotlib.pyplot as plt

@dataclass
class Wagon:
    length: float
    angle: float    # relative to the previous wagon (radians)
    fixed: bool = False

    def extend(self, delta: float):
        """Add (or subtract, if negative) length to this wagon, clamped to ≥0."""
        self.length = max(self.length + delta, 0.0)



class Microtubule:
    """
    Represents a single microtubule as a chain (“train”) of straight wagons (segments).
    """

    def __init__(
        self,
        base_point: np.ndarray,         # (x0, y0) pixel coordinates of “anchor”
        base_orientation: float,        # absolute orientation of first (fixed) wagon
        base_wagon_length: float,       # length of that fixed wagon
        profile: np.ndarray,            # length‐over‐time array (shape: num_frames)
        max_num_wagons: int,
        max_angle: float,
        angle_change_prob: float,
        min_wagon_length: float,
        max_wagon_length: float,
        instance_id: int = 0,
    ):
        # 1) Create the fixed base wagon
        self.wagons: List[Wagon] = [
            Wagon(length=base_wagon_length, angle=0.0, fixed=True)
        ]

        # 2) If the profile’s initial total length > base length, create a tip wagon:
        initial_total = profile[0]
        tip_length = max(initial_total - base_wagon_length, 0.0)
        if tip_length > 0:
            angle = 0.0
            if angle_change_prob > 0.0:
                angle = np.random.uniform(-max_angle, max_angle)
            self.wagons.append(Wagon(length=tip_length, angle=angle, fixed=False))

        # 3) Store parameters for later use:
        self.base_point = base_point.copy()
        self.base_orientation = base_orientation
        self.base_wagon_length = base_wagon_length
        self.profile = profile         # sawtooth or sinusoid length over time
        self.max_num_wagons = max_num_wagons
        self.max_angle = max_angle
        self.angle_change_prob = angle_change_prob
        self.min_wagon_length = min_wagon_length
        self.max_wagon_length = max_wagon_length

        # Convenience:
        self.current_length = sum(w.length for w in self.wagons)
        self.frame_idx = 0       # will be set externally each frame
        self.instance_id = instance_id     # for ground truth when drawing

    @property
    def total_length(self) -> float:
        return sum(w.length for w in self.wagons)


    def step_to_length(self, frame_idx: int):
        """
        Adjust wagons so that total_length == profile[frame_idx],
        preserving the angular chain for both shrinking and growing.
        """
        target_length = self.profile[frame_idx]

        # 1. The base wagon can shrink if the total length is less than its
        #    default length, but it grows back to its default length otherwise.
        self.wagons[0].length = min(self.base_wagon_length, target_length)

        # 2. The target length for the DYNAMIC part of the train is whatever
        #    remains after the base wagon is accounted for.
        target_dynamic_length = max(0.0, target_length - self.wagons[0].length)

        # 3. Get the current length of all dynamic wagons (index >= 1).
        current_dynamic_length = sum(w.length for w in self.wagons[1:])

        # 4. Calculate the change needed for the dynamic part.
        delta = target_dynamic_length - current_dynamic_length

        if abs(delta) < 1e-6:  # No significant change needed
            self.current_length = sum(w.length for w in self.wagons)
            return

        # ─── GROWING DYNAMIC PART ──────────────────────────────────────────
        if delta > 0:
            to_add = delta
            # First, re-inflate existing dynamic wagons that have shrunk.
            # This makes the microtubule "remember" its shape.
            for i in range(1, len(self.wagons)):
                wagon = self.wagons[i]
                space_available = self.max_wagon_length - wagon.length
                can_add = min(to_add, space_available)

                wagon.length += can_add
                to_add -= can_add

                if to_add <= 1e-6:
                    break

            # If there's still growth left, it means all existing wagons are full.
            # Now, and only now, we add new wagons at the tip.
            while to_add > 1e-6 and len(self.wagons) < self.max_num_wagons:
                angle = np.random.uniform(-self.max_angle, self.max_angle) if self.angle_change_prob > 0 else 0.0
                new_wagon_len = min(to_add, self.max_wagon_length)
                self.wagons.append(Wagon(length=new_wagon_len, angle=angle, fixed=False))
                to_add -= new_wagon_len

        # ─── SHRINKING DYNAMIC PART ────────────────────────────────────────
        elif delta < 0:
            to_remove = -delta
            # Shrink from the tip backwards. We set length to 0 but do not delete
            # the wagon, which preserves the angle for the next growth cycle.
            for i in range(len(self.wagons) - 1, 0, -1):
                wagon = self.wagons[i]
                length_to_remove_from_this_wagon = min(to_remove, wagon.length)

                wagon.length -= length_to_remove_from_this_wagon
                to_remove -= length_to_remove_from_this_wagon

                if to_remove <= 1e-6:
                    break

        # Update total length for convenience.
        self.current_length = sum(w.length for w in self.wagons)

    def maybe_rebend(self):
        """
        For each non‐fixed wagon (index ≥1), re-sample angle with prob=angle_change_prob
        """
        for i in range(1, len(self.wagons)):
            if np.random.random() < self.angle_change_prob:
                self.wagons[i].angle = np.random.uniform(-self.max_angle, self.max_angle)

    def draw(self, frame: np.ndarray, mask: np.ndarray, cfg: SyntheticDataConfig) -> list[dict]:
        """
        Rasterize each straight wagon, then apply fixed/moving spots, etc.
        Returns ground truth for each wagon.
        """
        # self.maybe_rebend()

        # Starting absolute angles and position:
        abs_angle = self.base_orientation
        abs_pos = self.base_point.copy()
        gt_info = []

        for idx, w in enumerate(self.wagons):
            abs_angle += w.angle
            dx = w.length * np.cos(abs_angle)
            dy = w.length * np.sin(abs_angle)
            new_pos = abs_pos + np.array([dx, dy], dtype=np.float32)

            # Draw Gaussian line from abs_pos → new_pos
            sigma_x = cfg.sigma_x * (1 + np.random.normal(0, cfg.width_var_std))
            sigma_y = cfg.sigma_y * (1 + np.random.normal(0, cfg.width_var_std))

            draw_gaussian_line(
                frame,
                mask,
                abs_pos, new_pos,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                contrast=cfg.tubulus_contrast,
                mask_idx=self.instance_id
            )

            gt_info.append({
                "frame_idx": cfg._frame_idx,
                "wagon_index": idx,
                "start": abs_pos.tolist(),
                "end": new_pos.tolist(),
                "angle": float(w.angle),
                "length": float(w.length),
                "instance_id": self.instance_id,
            })
            abs_pos = new_pos.copy()

        return gt_info

