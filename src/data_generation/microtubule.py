import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config.synthetic_data import SyntheticDataConfig
from data_generation.utils import draw_gaussian_line_rgb

logger = logging.getLogger(f"mt.{__name__}")


@dataclass
class Wagon:
    """Represents a visual segment of a microtubule for drawing purposes."""
    length: float
    angle: float  # relative to the previous wagon (radians)
    is_seed: bool = False


class Microtubule:
    """
    Represents a single microtubule with stateful, event-driven dynamics.
    The list of wagons is now persistent, creating a stable shape that grows/shrinks at the tip.
    """

    def __init__(self, cfg: SyntheticDataConfig, base_point: np.ndarray, instance_id: int = 0):
        logger.debug(f"Initializing Microtubule instance ID: {instance_id} at base point: {base_point.tolist()}.")
        self.cfg = cfg
        self.instance_id = instance_id
        self.frame_idx = 0

        # --- Initialize fixed properties ---
        self.base_point = base_point.copy()
        self.base_orientation = np.random.uniform(0.0, 2 * np.pi)
        seed_length = np.random.uniform(cfg.min_base_wagon_length, cfg.max_base_wagon_length)
        self.max_len = np.random.uniform(cfg.microtubule_length_min, cfg.microtubule_length_max)

        # --- Initialize stateful attributes ---
        self.state = np.random.choice(["growing", "shrinking"])
        self.min_pause_counter = 0

        # The list of wagons is the primary state for shape and length.
        self.seed_wagon = Wagon(length=seed_length, angle=0.0, is_seed=True)
        self.wagons: List[Wagon] = [self.seed_wagon]

        # <-- FIX: Bending properties MUST be initialized BEFORE the initial tail is built.
        # --- Initialize bending properties ---
        self.bending = np.random.random() < cfg.bending_prob
        self.max_angle = cfg.max_angle if self.bending else 0.0
        self.max_angle_sign_changes = cfg.max_angle_sign_changes
        self._current_bend_sign = np.random.choice([-1.0, 1.0])
        self._sign_changes_count = 0

        # Now, initialize to a random starting length by adding tail wagons.
        initial_length = np.random.uniform(seed_length, self.max_len)
        self._add_tail_length(initial_length - seed_length)

        logger.debug(
            f"MT {instance_id}: Initial state: '{self.state}', length: {self.total_length:.2f}, max_len: {self.max_len:.2f}.")

    @property
    def total_length(self) -> float:
        """Calculate total length by summing all wagon lengths."""
        return sum(w.length for w in self.wagons)

    def step(self):
        """
        Simulates one time step, updating state and modifying the wagon list at the tip.
        """
        self.frame_idx += 1
        prev_state = self.state

        # 1. Determine state transition logic
        if self.state == "growing":
            if self.total_length >= self.max_len:
                self.state = "shrinking"
            elif np.random.random() < self.cfg.catastrophe_prob:
                self.state = "shrinking"

        elif self.state == "shrinking":
            if self.total_length <= self.seed_wagon.length:
                self.state = "paused_min"
            elif np.random.random() < self.cfg.rescue_prob:
                self.state = "growing"

        elif self.state == "paused_min":
            self.min_pause_counter += 1
            if (np.random.random() < self.cfg.rescue_prob) or (
                    self.min_pause_counter >= self.cfg.max_pause_at_min_frames):
                self.state = "growing"
                self.min_pause_counter = 0

        # 2. Apply the change in length based on the NEW state.
        if self.state == 'growing':
            self._add_tail_length(self.cfg.growth_speed)
        elif self.state == 'shrinking':
            self._remove_tail_length(self.cfg.shrink_speed)

        # 3. Log state changes.
        if self.state != prev_state:
            logger.debug(
                f"MT {self.instance_id} Frame {self.frame_idx}: State changed from '{prev_state}' to '{self.state}'. Length: {self.total_length:.2f}")

    def _add_tail_length(self, amount_to_add: float):
        """Adds length to the tail, extending the last wagon or adding new ones."""
        if amount_to_add <= 0: return

        if len(self.wagons) <= 1:
            self.wagons.append(Wagon(length=0, angle=self._get_next_bend_angle()))

        last_wagon = self.wagons[-1]
        space_in_last_wagon = self.cfg.tail_wagon_length - last_wagon.length
        add_here = min(amount_to_add, space_in_last_wagon)
        last_wagon.length += add_here
        remaining_to_add = amount_to_add - add_here

        while remaining_to_add > 0:
            new_len = min(remaining_to_add, self.cfg.tail_wagon_length)
            self.wagons.append(Wagon(length=new_len, angle=self._get_next_bend_angle()))
            remaining_to_add -= new_len

    def _remove_tail_length(self, amount_to_remove: float):
        """Removes length from the tail by shrinking and popping wagons from the end."""
        if amount_to_remove <= 0: return

        while amount_to_remove > 0 and len(self.wagons) > 1:
            last_wagon = self.wagons[-1]
            removable_from_wagon = last_wagon.length

            if amount_to_remove >= removable_from_wagon:
                self.wagons.pop()
                amount_to_remove -= removable_from_wagon
            else:
                last_wagon.length -= amount_to_remove
                amount_to_remove = 0

    def _get_next_bend_angle(self) -> float:
        """Helper to get a bending angle for a new tail wagon."""
        if not self.bending: return 0.0

        can_flip = self._sign_changes_count < self.max_angle_sign_changes
        if can_flip and np.random.random() < self.cfg.prob_to_flip_bend:
            self._current_bend_sign *= -1.0
            self._sign_changes_count += 1

        return self._current_bend_sign * np.random.uniform(0, self.max_angle)

    def draw(
            self,
            frame: np.ndarray,
            microtubule_mask: Optional[np.ndarray],
            cfg: SyntheticDataConfig,
            seed_mask: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """Rasterizes the microtubule by drawing its persistent list of wagons."""
        logger.debug(f"MT {self.instance_id}: Drawing for frame {self.frame_idx}. Total wagons: {len(self.wagons)}.")

        abs_angle = self.base_orientation
        abs_pos = self.base_point.astype(np.float32)
        gt_info = []

        for idx, w in enumerate(self.wagons):
            if w.length <= 1e-6: continue

            abs_angle += w.angle
            dx = w.length * np.cos(abs_angle)
            dy = w.length * np.sin(abs_angle)
            new_pos = abs_pos + np.array([dx, dy], dtype=np.float32)

            sigma_x = max(0.0, cfg.sigma_x * (1 + np.random.normal(0, cfg.tubule_width_variation)))
            sigma_y = max(0.0, cfg.sigma_y * (1 + np.random.normal(0, cfg.tubule_width_variation)))

            base_contrast = cfg.tubulus_contrast
            is_tip_wagon = (idx == len(self.wagons) - 1)
            if self.state == "growing" and is_tip_wagon:
                base_contrast *= cfg.tip_brightness_factor

            r, g, b = base_contrast, base_contrast, base_contrast
            if w.is_seed:
                r += cfg.seed_red_channel_boost

            additional_mask_to_pass = (seed_mask if w.is_seed else None)

            try:
                draw_gaussian_line_rgb(
                    frame, microtubule_mask, abs_pos, new_pos,
                    sigma_x=sigma_x, sigma_y=sigma_y,
                    color_contrast_rgb=(r, g, b),
                    mask_idx=self.instance_id,
                    additional_mask=additional_mask_to_pass,
                )
            except Exception as e:
                logger.error(f"MT {self.instance_id}, Wagon {idx}: Failed to draw line: {e}", exc_info=True)

            gt_info.append({
                "frame_idx": self.frame_idx, "wagon_index": idx,
                "start": abs_pos.tolist(), "end": new_pos.tolist(),
                "length": float(w.length), "instance_id": self.instance_id,
            })
            abs_pos = new_pos.copy()

        return gt_info