from dataclasses import dataclass
from typing import List

import numpy as np

from config.synthetic_data import SyntheticDataConfig
from data_generation.utils import draw_gaussian_line_on_rgb


# HELPER FUNCTION to generate the length profile
def _generate_stochastic_profile(
        num_frames: int,
        min_len: float,
        max_len: float,
        growth_speed: float,
        shrink_speed: float,
        catastrophe_prob: float,
        rescue_prob: float,
        pause_on_min: int,
        pause_on_max: int
) -> np.ndarray:
    """Generates a length-over-time profile based on stochastic parameters."""
    profile = np.zeros(num_frames)
    state = "growing"
    current_length = min_len
    pause_counter = 0

    for i in range(num_frames):
        if state == "growing":
            if current_length >= max_len:
                state = "pausing_max"
                pause_counter = pause_on_max
            else:
                current_length += growth_speed
                if np.random.random() < catastrophe_prob:
                    state = "shrinking"

        elif state == "shrinking":
            if current_length <= min_len:
                state = "pausing_min"
                pause_counter = pause_on_min
            else:
                current_length -= shrink_speed
                if np.random.random() < rescue_prob:
                    state = "growing"

        elif state == "pausing_max":
            pause_counter -= 1
            if pause_counter <= 0:
                state = "shrinking"

        elif state == "pausing_min":
            pause_counter -= 1
            if pause_counter <= 0:
                state = "growing"

        profile[i] = np.clip(current_length, min_len, max_len)

    return profile


@dataclass
class Wagon:
    length: float
    angle: float  # relative to the previous wagon (radians)
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
            cfg: SyntheticDataConfig,
            base_point: np.ndarray,
            instance_id: int = 0,
    ):

        base_orientation = np.random.uniform(0.0, 2 * np.pi)
        base_wagon_length = np.random.uniform(
            cfg.min_base_wagon_length,
            cfg.max_base_wagon_length
        )

        # 1) Generate the length-over-time profile internally
        min_len = np.random.uniform(cfg.min_length_min, cfg.min_length_max)
        max_len = np.random.uniform(cfg.max_length_min, cfg.max_length_max)
        self.profile = _generate_stochastic_profile(
            num_frames=cfg.num_frames,
            min_len=min_len, max_len=max_len,
            growth_speed=cfg.growth_speed, shrink_speed=cfg.shrink_speed,
            catastrophe_prob=cfg.catastrophe_prob, rescue_prob=cfg.rescue_prob,
            pause_on_min=cfg.pause_on_min_length, pause_on_max=cfg.pause_on_max_length
        )

        # 2) Create the fixed base wagon
        self.wagons: List[Wagon] = [
            Wagon(length=base_wagon_length, angle=0.0, fixed=True)
        ]

        # 3) If the profile’s initial total length > base length, create a tip wagon
        initial_total = self.profile[0]
        tip_length = max(initial_total - base_wagon_length, 0.0)
        if tip_length > 0:
            angle = 0.0
            if cfg.max_angle_change_prob > 0.0:
                angle = np.random.uniform(-cfg.max_angle, cfg.max_angle)
            self.wagons.append(Wagon(length=tip_length, angle=angle, fixed=False))

        # 4) Store parameters and state
        self.base_point = base_point.copy()
        self.base_orientation = base_orientation
        self.base_wagon_length = base_wagon_length
        self.max_num_wagons = cfg.max_num_wagons
        self.max_angle = cfg.max_angle
        self.angle_change_prob = cfg.max_angle_change_prob
        self.min_wagon_length = np.random.uniform(cfg.min_wagon_length_min, cfg.min_wagon_length_max)
        self.max_wagon_length = np.random.uniform(cfg.max_wagon_length_min, cfg.max_wagon_length_max)

        # NEW: Store current state
        self.state = "growing"  # Initial state
        self.current_length = sum(w.length for w in self.wagons)
        self.frame_idx = 0
        self.instance_id = instance_id

    @property
    def total_length(self) -> float:
        return sum(w.length for w in self.wagons)

    def step_to_length(self, frame_idx: int):
        """
        Adjust wagons to match profile[frame_idx] and update the dynamic state.
        """
        target_length = self.profile[frame_idx]

        # NEW: Update the state based on length change
        if frame_idx > 0:
            if self.profile[frame_idx] > self.profile[frame_idx - 1]:
                self.state = "growing"
            elif self.profile[frame_idx] < self.profile[frame_idx - 1]:
                self.state = "shrinking"
            # If length is the same, state remains unchanged (pausing)
        else:
            self.state = "growing"  # Start by growing

        # ... (the rest of the logic is the same as before) ...
        self.wagons[0].length = min(self.base_wagon_length, target_length)
        target_dynamic_length = max(0.0, target_length - self.wagons[0].length)
        current_dynamic_length = sum(w.length for w in self.wagons[1:])
        delta = target_dynamic_length - current_dynamic_length

        if abs(delta) < 1e-6:
            self.current_length = sum(w.length for w in self.wagons)
            return

        if delta > 0:
            to_add = delta
            for i in range(1, len(self.wagons)):
                wagon = self.wagons[i]
                space_available = self.max_wagon_length - wagon.length
                can_add = min(to_add, space_available)
                wagon.length += can_add
                to_add -= can_add
                if to_add <= 1e-6: break

            while to_add > 1e-6 and len(self.wagons) < self.max_num_wagons:
                angle = np.random.uniform(-self.max_angle, self.max_angle) if self.angle_change_prob > 0 else 0.0
                new_wagon_len = min(to_add, self.max_wagon_length)
                self.wagons.append(Wagon(length=new_wagon_len, angle=angle, fixed=False))
                to_add -= new_wagon_len
        elif delta < 0:
            to_remove = -delta
            for i in range(len(self.wagons) - 1, 0, -1):
                wagon = self.wagons[i]
                length_to_remove_from_this_wagon = min(to_remove, wagon.length)
                wagon.length -= length_to_remove_from_this_wagon
                to_remove -= length_to_remove_from_this_wagon
                if to_remove <= 1e-6: break

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
        Rasterizes each wagon using the new flexible contrast model.
        """
        abs_angle = self.base_orientation
        abs_pos = self.base_point.copy()
        gt_info = []

        for idx, w in enumerate(self.wagons):
            abs_angle += w.angle
            dx = w.length * np.cos(abs_angle)
            dy = w.length * np.sin(abs_angle)
            new_pos = abs_pos + np.array([dx, dy], dtype=np.float32)

            sigma_x = cfg.sigma_x * (1 + np.random.normal(0, cfg.width_var_std))
            sigma_y = cfg.sigma_y * (1 + np.random.normal(0, cfg.width_var_std))

            # --- NEW LOGIC: Calculate per-channel contrast ---

            # 1. Start with the base contrast for all channels.
            #    This can be positive (bright) or negative (dark).
            base_contrast = cfg.tubulus_contrast

            # 2. Apply tip brightness factor if applicable.
            is_tip_wagon = (idx == len(self.wagons) - 1)
            if self.state == "growing" and is_tip_wagon:
                base_contrast *= cfg.tip_brightness_factor

            # 3. Create the RGB contrast tuple.
            r_contrast = base_contrast
            g_contrast = base_contrast
            b_contrast = base_contrast

            # 4. For seed segments, add the red channel boost.
            is_seed_wagon = (idx == 0)
            if is_seed_wagon:
                r_contrast += cfg.seed_red_channel_boost

            color_contrast_rgb = (r_contrast, g_contrast, b_contrast)

            # 5. Call the drawing function with the final calculated values.
            draw_gaussian_line_on_rgb(
                frame, mask, abs_pos, new_pos,
                sigma_x=sigma_x, sigma_y=sigma_y,
                color_contrast_rgb=color_contrast_rgb,
                mask_idx=self.instance_id
            )

            # Ground truth generation
            gt_info.append({
                "frame_idx": -1, # Will be set later
                "wagon_index": idx,
                "start": abs_pos.tolist(),
                "end": new_pos.tolist(),
                "angle": float(w.angle),
                "length": float(w.length),
                "instance_id": self.instance_id,
            })
            abs_pos = new_pos.copy()

        return gt_info