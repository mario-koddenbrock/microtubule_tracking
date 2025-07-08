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
    Growth and shrinkage occur at the tip, governed by catastrophe/rescue probabilities.
    """

    def __init__(self, cfg: SyntheticDataConfig, base_point: np.ndarray, instance_id: int = 0):
        logger.debug(f"Initializing Microtubule instance ID: {instance_id} at base point: {base_point.tolist()}.")
        self.cfg = cfg
        self.instance_id = instance_id
        self.frame_idx = 0

        # --- Initialize fixed properties of this specific microtubule ---
        self.base_point = base_point.copy()
        self.base_orientation = np.random.uniform(0.0, 2 * np.pi)
        seed_length = np.random.uniform(cfg.min_base_wagon_length, cfg.max_base_wagon_length)
        self.max_len = np.random.uniform(cfg.max_length_min, cfg.max_length_max)

        self.seed_wagon = Wagon(length=seed_length, angle=0.0, is_seed=True)
        self.wagons: List[Wagon] = [self.seed_wagon]

        # --- Initialize dynamic state ---
        self.state = np.random.choice(["growing", "shrinking"])

        # <-- NEW: Start with a realistic, random length, not just the seed length.
        initial_length = np.random.uniform(seed_length, self.max_len)
        self.current_length = initial_length

        # <-- NEW: Counter for the pause at minimum length.
        self.min_pause_counter = 0

        logger.debug(
            f"MT {instance_id}: Initial state: '{self.state}', length: {self.current_length:.2f}, max_len: {self.max_len:.2f}.")

        # --- Initialize bending properties for the tail ---
        self.bending = np.random.random() < cfg.bending_prob
        self.max_angle = cfg.max_angle if self.bending else 0.0
        self.max_angle_sign_changes = cfg.max_angle_sign_changes
        self._current_bend_sign = np.random.choice([-1.0, 1.0])
        self._sign_changes_count = 0

    def step(self):
        """
        Simulates one time step of microtubule dynamics, updating its state and length.
        """
        self.frame_idx += 1
        prev_state = self.state

        if self.state == "growing":
            self.current_length += self.cfg.growth_speed
            if self.current_length >= self.max_len:
                self.current_length = self.max_len
                self.state = "shrinking"
            elif np.random.random() < self.cfg.catastrophe_prob:
                self.state = "shrinking"

        elif self.state == "shrinking":
            self.current_length -= self.cfg.shrink_speed
            if self.current_length <= self.seed_wagon.length:
                self.current_length = self.seed_wagon.length
                self.state = "paused_min"
                self.min_pause_counter = 0  # <-- NEW: Reset pause counter upon entering state.
            elif np.random.random() < self.cfg.rescue_prob:
                self.state = "growing"

        elif self.state == "paused_min":
            # <-- NEW: Check for two escape conditions: a random rescue or hitting the max pause time.
            self.min_pause_counter += 1
            rescue_event = np.random.random() < self.cfg.rescue_prob
            max_pause_reached = self.min_pause_counter >= self.cfg.max_pause_at_min_frames

            if rescue_event or max_pause_reached:
                self.state = "growing"

        if self.state != prev_state:
            logger.debug(
                f"MT {self.instance_id} Frame {self.frame_idx}: State changed from '{prev_state}' to '{self.state}'. Length: {self.current_length:.2f}")

    def _update_wagons_for_drawing(self):
        """
        Translates the current_length into a list of "wagon" segments for drawing.
        """
        tail_length = self.current_length - self.seed_wagon.length
        self.wagons = [self.seed_wagon]

        if tail_length <= 1e-6:
            return

        self._sign_changes_count = 0
        num_full_wagons = int(tail_length // self.cfg.tail_wagon_length)
        last_wagon_len = tail_length % self.cfg.tail_wagon_length

        for _ in range(num_full_wagons):
            self.wagons.append(Wagon(length=self.cfg.tail_wagon_length, angle=self._get_next_bend_angle()))

        if last_wagon_len > 1e-6:
            self.wagons.append(Wagon(length=last_wagon_len, angle=self._get_next_bend_angle()))

    def _get_next_bend_angle(self) -> float:
        """Helper to get a bending angle for a new tail wagon."""
        if not self.bending:
            return 0.0

        can_flip = self._sign_changes_count < self.max_angle_sign_changes
        if can_flip and np.random.random() < self.cfg.prob_to_flip_bend:
            self._current_bend_sign *= -1.0
            self._sign_changes_count += 1

        return self._current_bend_sign * np.random.uniform(0, self.max_angle)

    def draw(
            self,
            frame: np.ndarray,
            tubuli_mask: Optional[np.ndarray],
            cfg: SyntheticDataConfig,
            seed_mask: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """Rasterizes the microtubule based on its current state."""
        self._update_wagons_for_drawing()

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
                    frame, tubuli_mask, abs_pos, new_pos,
                    sigma_x=sigma_x, sigma_y=sigma_y,
                    color_contrast_rgb=(r, g, b),
                    mask_idx=self.instance_id,
                    additional_mask=additional_mask_to_pass,
                )
            except Exception as e:
                logger.error(f"MT {self.instance_id}, Wagon {idx}: Failed to draw line: {e}", exc_info=True)

            gt_info.append({
                "frame_idx": self.frame_idx,
                "wagon_index": idx,
                "start": abs_pos.tolist(),
                "end": new_pos.tolist(),
                "length": float(w.length),
                "instance_id": self.instance_id,
            })
            abs_pos = new_pos.copy()

        return gt_info