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
        never touching the fixed base wagon (index 0).
        """
        target_length = self.profile[frame_idx]
        current_length = sum(w.length for w in self.wagons)
        delta_length = target_length - current_length

        # If only the base wagon exists, and we need to grow:
        if len(self.wagons) == 1 and target_length > self.wagons[0].length:
            new_len = target_length - self.wagons[0].length
            # sample a random angle if allowed
            angle = 0.0
            if self.angle_change_prob > 0.0:
                angle = np.random.uniform(-self.max_angle, self.max_angle)
            self.wagons.append(Wagon(length=new_len, angle=angle, fixed=False))
            self.current_length = target_length
            return

        # Otherwise, distribute delta_length among dynamic wagons (indices ≥ 1)
        remainder = delta_length

        # ─── GROWING ─────────────────────────────────────────────────────────
        if delta_length > 0:
            idx = len(self.wagons) - 1  # start with the tip wagon
            while remainder > 0 and idx >= 1:
                w = self.wagons[idx]
                w.length += remainder
                # If it exceeds max_wagon_length and can split:
                if w.length > self.max_wagon_length and len(self.wagons) < self.max_num_wagons:
                    excess = w.length - self.max_wagon_length
                    w.length = self.max_wagon_length
                    # create a new wagon for the excess
                    angle = 0.0
                    if self.angle_change_prob > 0.0:
                        angle = np.random.uniform(-self.max_angle, self.max_angle)
                    self.wagons.append(Wagon(length=excess, angle=angle, fixed=False))
                    remainder = excess
                    idx = len(self.wagons) - 1
                else:
                    remainder = 0.0
                # If we still have remainder but cannot add more wagons (we’re at max):
                if remainder > 0 and idx == 1 and len(self.wagons) == self.max_num_wagons:
                    # clamp the tip beyond max if needed
                    w.length += remainder
                    remainder = 0.0
                idx -= 1

        # ─── SHRINKING ────────────────────────────────────────────────────────
        elif delta_length < 0:
            remainder = -delta_length  # positive amount to remove
            idx = len(self.wagons) - 1
            while remainder > 0 and idx >= 1:
                w = self.wagons[idx]
                w.length -= remainder
                if w.length < self.min_wagon_length and idx > 1:
                    deficit = self.min_wagon_length - w.length
                    # remove this entire wagon
                    self.wagons.pop(idx)
                    idx -= 1
                    self.wagons[idx].length -= deficit
                    remainder = deficit
                else:
                    if w.length < 0:
                        w.length = 0.0
                    remainder = 0.0
                idx -= 1

            # If we still have “too much shrink” and only base remains:
            if remainder > 0:
                # collapse everything except the base
                self.wagons = [self.wagons[0]]
                self.wagons[0].length = self.profile[frame_idx]
        # Update total
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

            frame, mask = draw_gaussian_line(
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

