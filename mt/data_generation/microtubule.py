import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import numpy as np

from mt.config.synthetic_data import SyntheticDataConfig
from mt.data_generation.utils import draw_gaussian_line_rgb

logger = logging.getLogger(f"mt.{__name__}")


class MicrotubuleState(Enum):
    GROWING = auto()
    SHRINKING = auto()
    PAUSED = auto()


@dataclass
class Wagon:
    """Represents a single, persistent segment of a microtubule."""
    length: float
    is_seed: bool = False
    relative_bend_angle: float = 0.0  # Angle relative to the previous wagon


class Microtubule:
    """
    Represents a single microtubule with stateful, event-driven dynamics.
    It consists of a persistent 'base' (seed) and a dynamic 'tail'.
    """

    def __init__(self, cfg: SyntheticDataConfig, base_point: np.ndarray, instance_id: int = 0):
        self.instance_id = instance_id
        self.base_point = base_point.astype(np.float32)
        self.base_orientation = np.random.uniform(0, 2 * np.pi)
        self.frame_idx = 0
        self.state = MicrotubuleState.GROWING
        self.pause_frames_at_min = 0

        # Initialize the persistent base wagon (the "seed")
        base_length = np.random.uniform(cfg.base_wagon_length_min, cfg.base_wagon_length_max)
        self.wagons: List[Wagon] = [Wagon(length=base_length, is_seed=True)]

        # Bending state must be initialized before adding the tail
        self.bend_angle_sign_changes_left = cfg.max_angle_sign_changes
        self.current_bend_sign = np.random.choice([-1, 1])

        # Initialize the dynamic tail length
        initial_tail_length = np.random.uniform(
            cfg.microtubule_length_min - base_length,
            cfg.microtubule_length_max - base_length
        )
        self._add_tail_length(initial_tail_length, cfg)

        # Minus-end state
        self.minus_end_target_length = max(0.0, np.random.normal(
            cfg.minus_end_target_length_mean, cfg.minus_end_target_length_std
        ))
        # Start with a random length between 0 and its target length
        self.minus_end_length = np.random.uniform(0, self.minus_end_target_length)

        logger.debug(
            f"MT {self.instance_id} created. Base len: {base_length:.2f}, "
            f"Initial tail: {initial_tail_length:.2f}, Total: {self.total_length:.2f}, "
            f"Minus-end target: {self.minus_end_target_length:.2f}"
        )

    @property
    def total_length(self) -> float:
        """Calculates the total length of all wagons."""
        return sum(w.length for w in self.wagons)

    def step(self, cfg: SyntheticDataConfig):
        """
        Advances the microtubule simulation by one time step, updating its state and length.
        """
        self.frame_idx += 1

        # 1. Update state based on catastrophe/rescue probabilities
        if self.state == MicrotubuleState.GROWING:
            if np.random.rand() < cfg.catastrophe_prob:
                self.state = MicrotubuleState.SHRINKING
                logger.debug(f"MT {self.instance_id}: Catastrophe -> SHRINKING")
        elif self.state == MicrotubuleState.SHRINKING:
            if self.total_length <= self.wagons[0].length:  # At min length (only seed left)
                self.state = MicrotubuleState.PAUSED
                self.pause_frames_at_min = 0
                logger.debug(f"MT {self.instance_id}: Min length reached -> PAUSED")
            elif np.random.rand() < cfg.rescue_prob:
                self.state = MicrotubuleState.GROWING
                logger.debug(f"MT {self.instance_id}: Rescue -> GROWING")

        # 2. Update plus-end tail length based on the current state
        if self.state == MicrotubuleState.GROWING:
            growth = np.random.normal(cfg.growth_speed, cfg.growth_speed * 0.1)
            self._add_tail_length(growth, cfg)
        elif self.state == MicrotubuleState.SHRINKING:
            shrinkage = np.random.normal(cfg.shrink_speed, cfg.shrink_speed * 0.1)
            if self._remove_tail_length(shrinkage):
                self.state = MicrotubuleState.PAUSED  # Tail is gone
                self.pause_frames_at_min = 0
        elif self.state == MicrotubuleState.PAUSED:
            self.pause_frames_at_min += 1
            if self.pause_frames_at_min >= cfg.max_pause_at_min_frames:
                self.state = MicrotubuleState.GROWING  # Forced rescue
                logger.debug(f"MT {self.instance_id}: Forced rescue from PAUSED -> GROWING")

        # 3. Update minus-end tail length
        if abs(self.minus_end_length - self.minus_end_target_length) > 1e-6:
            direction = np.sign(self.minus_end_target_length - self.minus_end_length)
            change = min(cfg.minus_end_velocity, abs(self.minus_end_target_length - self.minus_end_length))
            self.minus_end_length += direction * change

    def _add_tail_length(self, length_to_add: float, cfg: SyntheticDataConfig):
        """Adds length to the dynamic tail, creating new visual wagons if needed."""
        if not self.wagons: return
        last_wagon = self.wagons[-1]

        if last_wagon.is_seed:
            # Start a new dynamic wagon
            new_wagon = Wagon(length=length_to_add, relative_bend_angle=self._get_next_bend_angle(cfg))
            self.wagons.append(new_wagon)
        else:
            # Add to the last dynamic wagon
            last_wagon.length += length_to_add

        # Subdivide the last wagon if it's too long
        while len(self.wagons) > 1 and not self.wagons[-1].is_seed and self.wagons[-1].length > cfg.tail_wagon_length:
            last = self.wagons[-1]
            excess_length = last.length - cfg.tail_wagon_length
            last.length = cfg.tail_wagon_length
            new_wagon = Wagon(length=excess_length, relative_bend_angle=self._get_next_bend_angle(cfg))
            self.wagons.append(new_wagon)

    def _remove_tail_length(self, length_to_remove: float) -> bool:
        """Removes length from the dynamic tail. Returns True if tail is gone."""
        while length_to_remove > 0 and len(self.wagons) > 1:
            last_wagon = self.wagons[-1]
            if last_wagon.is_seed: break  # Should not happen if len > 1

            if length_to_remove >= last_wagon.length:
                length_to_remove -= last_wagon.length
                self.wagons.pop()
            else:
                last_wagon.length -= length_to_remove
                length_to_remove = 0
        return len(self.wagons) == 1

    def _get_next_bend_angle(self, cfg: SyntheticDataConfig) -> float:
        """Determines the bend angle for a new wagon based on a gamma distribution."""
        # If scale is zero, no bending will occur.
        if cfg.bending_angle_gamma_scale == 0:
            return 0.0

        # Flip sign probabilistically if allowed
        if self.bend_angle_sign_changes_left > 0 and np.random.rand() < cfg.prob_to_flip_bend:
            self.current_bend_sign *= -1
            self.bend_angle_sign_changes_left -= 1

        angle = np.random.gamma(cfg.bending_angle_gamma_shape, cfg.bending_angle_gamma_scale)
        return self.current_bend_sign * angle

    def draw(
            self,
            frame: np.ndarray,
            mt_mask: Optional[np.ndarray],
            cfg: SyntheticDataConfig,
            seed_mask: Optional[np.ndarray] = None,
    ) -> List[dict]:
        logger.debug(
            f"MT {self.instance_id}: Drawing for {self.frame_idx}. Total wagons: {len(self.wagons)}.")

        abs_angle = self.base_orientation
        abs_pos = self.base_point.astype(np.float32)
        gt_info = []

        # --- Draw Minus-End (if it exists) ---
        if self.minus_end_length > 1e-6:
            minus_end_vec = np.array([np.cos(abs_angle + np.pi), np.sin(abs_angle + np.pi)])
            minus_end_pos = abs_pos + minus_end_vec * self.minus_end_length
            color_contrast = (cfg.tubulus_contrast, cfg.tubulus_contrast, cfg.tubulus_contrast)
            draw_gaussian_line_rgb(
                frame, mt_mask, abs_pos, minus_end_pos,
                cfg.psf_sigma_h, cfg.psf_sigma_v, color_contrast, self.instance_id
            )
            gt_info.append({
                "instance_id": self.instance_id,
                "segment_id": f"{self.instance_id}-M",
                "type": "minus_end",
                "start_pos": abs_pos.tolist(),
                "end_pos": minus_end_pos.tolist(),
                "length": self.minus_end_length,
            })

        # --- Draw Main Body (Plus-End and Seed) ---
        for idx, w in enumerate(self.wagons):
            start_pos = abs_pos.copy()
            abs_angle += w.relative_bend_angle
            vec = np.array([np.cos(abs_angle), np.sin(abs_angle)])
            end_pos = start_pos + vec * w.length
            abs_pos = end_pos
            psf_h = cfg.psf_sigma_h * (
                        1 + np.random.uniform(-cfg.tubule_width_variation, cfg.tubule_width_variation))
            psf_v = cfg.psf_sigma_v
            base_contrast = cfg.tubulus_contrast
            tip_brightness = cfg.tip_brightness_factor if self.state == MicrotubuleState.GROWING and idx == len(
                self.wagons) - 1 else 1.0
            if w.is_seed:
                color_contrast = (
                    base_contrast + cfg.seed_red_channel_boost,
                    base_contrast - cfg.seed_red_channel_boost,
                    base_contrast - cfg.seed_red_channel_boost
                )
                segment_type = "seed"
                current_mask = seed_mask
            else:
                color_contrast = (
                    base_contrast * tip_brightness,
                    base_contrast * tip_brightness,
                    base_contrast * tip_brightness
                )
                segment_type = "plus_end"
                current_mask = None
            draw_gaussian_line_rgb(
                frame, mt_mask, start_pos, end_pos,
                psf_h, psf_v, color_contrast, self.instance_id, additional_mask=current_mask
            )
            gt_info.append({
                "instance_id": self.instance_id,
                "segment_id": f"{self.instance_id}-{idx}",
                "type": segment_type,
                "start_pos": start_pos.tolist(),
                "end_pos": end_pos.tolist(),
                "length": w.length,
            })
        return gt_info
