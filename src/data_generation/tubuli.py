import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config.synthetic_data import SyntheticDataConfig
from data_generation.utils import draw_gaussian_line_rgb  

logger = logging.getLogger(f"mt.{__name__}")


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
        pause_on_max: int,
) -> np.ndarray:
    """
    Generates a length-over-time profile based on stochastic parameters.
    """
    logger.info(f"Generating stochastic length profile for {num_frames} frames.")
    logger.debug(
        f"Parameters: min_len={min_len:.2f}, max_len={max_len:.2f}, growth_speed={growth_speed:.2f}, shrink_speed={shrink_speed:.2f}, catastrophe_prob={catastrophe_prob:.4f}, rescue_prob={rescue_prob:.4f}, pause_on_min={pause_on_min}, pause_on_max={pause_on_max}.")

    profile = np.zeros(num_frames)

    # The initial state is chosen randomly
    state = np.random.choice(["growing", "shrinking"])
    current_length = np.random.uniform(min_len, max_len)  # Initialize within bounds

    logger.debug(f"Initial state: '{state}', initial length: {current_length:.2f}.")

    pause_counter = 0

    for i in range(num_frames):
        prev_state = state
        if state == "growing":
            if current_length >= max_len:
                state = "pausing_max"
                pause_counter = pause_on_max
                logger.debug(f"Frame {i}: Reached max_len ({max_len:.2f}). Transition to 'pausing_max'.")
            else:
                current_length += growth_speed
                if np.random.random() < catastrophe_prob:
                    state = "shrinking"
                    logger.debug(f"Frame {i}: Catastrophe occurred. Transition to 'shrinking'.")

        elif state == "shrinking":
            if current_length <= min_len:
                state = "pausing_min"
                pause_counter = pause_on_min
                logger.debug(f"Frame {i}: Reached min_len ({min_len:.2f}). Transition to 'pausing_min'.")
            else:
                current_length -= shrink_speed
                if np.random.random() < rescue_prob:
                    state = "growing"
                    logger.debug(f"Frame {i}: Rescue occurred. Transition to 'growing'.")

        elif state == "pausing_max":
            pause_counter -= 1
            if pause_counter <= 0:
                state = "shrinking"
                logger.debug(f"Frame {i}: Pause at max length ended. Transition to 'shrinking'.")

        elif state == "pausing_min":
            pause_counter -= 1
            if pause_counter <= 0:
                state = "growing"
                logger.debug(f"Frame {i}: Pause at min length ended. Transition to 'growing'.")

        if state != prev_state:
            logger.debug(
                f"Frame {i}: State changed from '{prev_state}' to '{state}'. Current length: {current_length:.2f}.")

        profile[i] = np.clip(current_length, min_len, max_len)
        logger.debug(f"Frame {i}: Final length for frame: {profile[i]:.2f}.")

    logger.info(
        f"Length profile generated. First 5 values: {profile[:5].round(2)}, Last 5 values: {profile[-5:].round(2)}.")
    return profile


@dataclass
class Wagon:
    length: float
    angle: float  # relative to the previous wagon (radians)
    fixed: bool = False

    def extend(self, delta: float):
        """Add (or subtract, if negative) length to this wagon, clamped to ≥0."""
        old_length = self.length
        self.length = max(self.length + delta, 0.0)
        logger.debug(f"Wagon extended: old_length={old_length:.2f}, delta={delta:.2f}, new_length={self.length:.2f}.")


class Microtubule:
    """
    Represents a single microtubule as a chain of wagons with stateful bending.
    """

    def __init__(
            self,
            cfg: SyntheticDataConfig,
            base_point: np.ndarray,
            instance_id: int = 0,
    ):
        logger.info(f"Initializing Microtubule instance ID: {instance_id} at base point: {base_point.tolist()}.")
        self.cfg = cfg  # Store cfg for easy access to parameters

        base_orientation = np.random.uniform(0.0, 2 * np.pi)
        base_wagon_length = np.random.uniform(cfg.min_base_wagon_length, cfg.max_base_wagon_length)
        logger.debug(
            f"MT {instance_id}: Base orientation={math.degrees(base_orientation):.2f} deg, base wagon length={base_wagon_length:.2f}.")

        # 1) Generate the length-over-time profile internally.
        min_len = np.random.uniform(cfg.min_length_min, cfg.min_length_max)
        max_len = np.random.uniform(cfg.max_length_min, cfg.max_length_max)
        logger.debug(f"MT {instance_id}: Generating length profile with min_len={min_len:.2f}, max_len={max_len:.2f}.")
        self.profile = _generate_stochastic_profile(
            num_frames=cfg.num_frames,
            min_len=min_len,
            max_len=max_len,
            growth_speed=cfg.growth_speed,
            shrink_speed=cfg.shrink_speed,
            catastrophe_prob=cfg.catastrophe_prob,
            rescue_prob=cfg.rescue_prob,
            pause_on_min=cfg.pause_on_min_length,
            pause_on_max=cfg.pause_on_max_length,
        )

        # 2) Create the fixed base wagon
        self.wagons: List[Wagon] = [Wagon(length=base_wagon_length, angle=0.0, fixed=True)]
        logger.debug(f"MT {instance_id}: Created base wagon with length {base_wagon_length:.2f}.")

        # 4) Store parameters and state
        self.base_point = base_point.copy()
        self.base_orientation = base_orientation
        self.base_wagon_length = base_wagon_length  # This fixed length is for the first wagon
        self.max_num_wagons = cfg.max_num_wagons
        self.bending = np.random.random() < cfg.bending_prob
        self.max_angle = cfg.max_angle if self.bending else 0.0
        self.min_wagon_length = np.random.uniform(
            cfg.min_wagon_length_min, cfg.min_wagon_length_max
        )
        self.max_wagon_length = np.random.uniform(
            cfg.max_wagon_length_min, cfg.max_wagon_length_max
        )

        self.max_angle_sign_changes = cfg.max_angle_sign_changes
        self.prob_to_flip_bend = cfg.prob_to_flip_bend
        self._current_bend_sign = np.random.choice([-1.0, 1.0])
        self._sign_changes_count = 0

        self.state = "initial"  # Will be updated in step_to_length
        self.current_length = sum(w.length for w in self.wagons)
        self.frame_idx = 0  # Current frame index the MT state represents
        self.instance_id = instance_id

        logger.debug(
            f"MT {instance_id}: Bending enabled: {self.bending}, max_angle={math.degrees(self.max_angle):.2f} deg, min_wagon_len={self.min_wagon_length:.2f}, max_wagon_len={self.max_wagon_length:.2f}.")

        # 3) Initialize the microtubule to its starting length from the profile
        initial_total = self.profile[0]
        initial_delta = initial_total - self.current_length
        if initial_delta > 0:
            logger.debug(
                f"MT {instance_id}: Initializing to profile length {initial_total:.2f}. Adding {initial_delta:.2f} length.")
            self._add_new_wagons(initial_delta)
        else:
            logger.debug(
                f"MT {instance_id}: Initial length {self.current_length:.2f} is already at or above profile start {initial_total:.2f}. No initial growth.")

        self.current_length = sum(w.length for w in self.wagons)  # Update after initial growth
        logger.info(f"Microtubule {instance_id} initialized with total length: {self.current_length:.2f}.")

    @property
    def total_length(self) -> float:
        return sum(w.length for w in self.wagons)

    def _add_new_wagons(self, amount_to_add: float):
        """A helper function that adds new wagons with controlled bending."""
        logger.debug(f"MT {self.instance_id}: Adding new wagons, total amount to add: {amount_to_add:.2f}.")
        to_add = amount_to_add
        initial_wagon_count = len(self.wagons)

        while to_add > 1e-6 and len(self.wagons) < self.max_num_wagons:
            can_flip = self._sign_changes_count < self.max_angle_sign_changes
            if can_flip and np.random.random() < self.cfg.prob_to_flip_bend:
                self._current_bend_sign *= -1.0
                self._sign_changes_count += 1
                logger.debug(
                    f"MT {self.instance_id}: Bending sign flipped. New sign: {self._current_bend_sign}, changes count: {self._sign_changes_count}.")

            angle_magnitude = np.random.uniform(0, self.max_angle)
            angle = self._current_bend_sign * angle_magnitude

            new_wagon_len = min(to_add, self.max_wagon_length)
            if new_wagon_len < self.min_wagon_length:
                new_wagon_len = self.min_wagon_length  # Ensure minimum wagon length if possible
                logger.debug(
                    f"MT {self.instance_id}: New wagon length adjusted to min_wagon_length {new_wagon_len:.2f}.")

            self.wagons.append(Wagon(length=new_wagon_len, angle=angle, fixed=False))
            to_add -= new_wagon_len
            logger.debug(
                f"MT {self.instance_id}: Added wagon {len(self.wagons) - 1} (length: {new_wagon_len:.2f}, angle: {math.degrees(angle):.2f} deg). Remaining to add: {to_add:.2f}.")

        if len(self.wagons) >= self.max_num_wagons and to_add > 1e-6:
            logger.warning(
                f"MT {self.instance_id}: Reached max_num_wagons ({self.max_num_wagons}). {to_add:.2f} length could not be added.")

        logger.debug(
            f"MT {self.instance_id}: Finished adding wagons. Total new wagons added: {len(self.wagons) - initial_wagon_count}.")

    def step_to_length(self, frame_idx: int):
        """
        Adjust wagons to match profile[frame_idx] and update the dynamic state.
        This method manages growth/shrinkage by modifying wagon lengths.
        """
        self.frame_idx = frame_idx
        target_length = self.profile[frame_idx]

        logger.debug(
            f"MT {self.instance_id}: Stepping to frame {frame_idx}. Target length: {target_length:.2f}, Current total length: {self.total_length:.2f}.")

        # Determine state based on profile trend
        prev_state = self.state
        if frame_idx > 0:
            if self.profile[frame_idx] > self.profile[frame_idx - 1]:
                self.state = "growing"
            elif self.profile[frame_idx] < self.profile[frame_idx - 1]:
                self.state = "shrinking"
            else:
                self.state = "pausing"
        else:  # For the very first frame
            if len(self.profile) > 1 and self.profile[1] > self.profile[0]:
                self.state = "growing"
            elif len(self.profile) > 1 and self.profile[1] < self.profile[0]:
                self.state = "shrinking"
            else:
                self.state = "pausing"

        if self.state != prev_state:
            logger.debug(f"MT {self.instance_id}: State changed from '{prev_state}' to '{self.state}'.")

        # Adjust fixed base wagon length (clamped to its original length or target_length)
        # The base wagon should ideally be fixed, but its length can be limited by target_length
        # to prevent it from being longer than the microtubule if target_length is very small.
        initial_base_len = self.wagons[0].length  # Store for comparison
        self.wagons[0].length = min(self.base_wagon_length,
                                    target_length)  # ensure base wagon doesn't exceed overall length
        if initial_base_len != self.wagons[0].length:
            logger.debug(
                f"MT {self.instance_id}: Base wagon length adjusted from {initial_base_len:.2f} to {self.wagons[0].length:.2f}.")

        target_dynamic_length = max(0.0, target_length - self.wagons[0].length)
        current_dynamic_length = sum(w.length for w in self.wagons[1:])
        delta = target_dynamic_length - current_dynamic_length

        logger.debug(
            f"MT {self.instance_id}: Target dynamic length: {target_dynamic_length:.2f}, Current dynamic length: {current_dynamic_length:.2f}, Delta: {delta:.2f}.")

        if abs(delta) < 1e-6:  # Very small change, almost no change needed
            self.current_length = self.total_length
            logger.debug(f"MT {self.instance_id}: Delta is negligible ({delta:.6f}). No length adjustment needed.")
            return

        if delta > 0:  # Need to grow
            to_add = delta
            logger.debug(f"MT {self.instance_id}: Growing. Amount to add: {to_add:.2f}.")
            for i in range(1, len(self.wagons)):
                wagon = self.wagons[i]
                space_available = self.max_wagon_length - wagon.length
                can_add = min(to_add, space_available)
                wagon.length += can_add
                to_add -= can_add
                logger.debug(
                    f"MT {self.instance_id}: Added {can_add:.2f} to wagon {i}. Remaining to add: {to_add:.2f}.")
                if to_add <= 1e-6:
                    break

            if to_add > 1e-6:  # Still length to add after filling existing wagons
                logger.debug(f"MT {self.instance_id}: Remaining length ({to_add:.2f}) added by creating new wagons.")
                self._add_new_wagons(to_add)

        elif delta < 0:  # Need to shrink
            to_remove = -delta
            logger.debug(f"MT {self.instance_id}: Shrinking. Amount to remove: {to_remove:.2f}.")
            # Iterate backwards to remove from the end first
            wagons_to_keep = [self.wagons[0]]  # Always keep the fixed base wagon
            for i in range(len(self.wagons) - 1, 0, -1):
                wagon = self.wagons[i]
                length_to_remove_from_this_wagon = min(to_remove, wagon.length)
                wagon.length -= length_to_remove_from_this_wagon
                to_remove -= length_to_remove_from_this_wagon
                logger.debug(
                    f"MT {self.instance_id}: Removed {length_to_remove_from_this_wagon:.2f} from wagon {i}. Remaining to remove: {to_remove:.2f}.")

                if wagon.length > 1e-6:  # If wagon still has length, keep it
                    wagons_to_keep.insert(1, wagon)  # Insert at beginning (after base) to maintain order
                else:
                    logger.debug(f"MT {self.instance_id}: Wagon {i} fully removed (length became zero).")

                if to_remove <= 1e-6:
                    break

            self.wagons = wagons_to_keep  # Update the list of wagons
            if to_remove > 1e-6:
                logger.warning(
                    f"MT {self.instance_id}: Still {to_remove:.2f} length to remove after removing all dynamic wagons. This should not happen if base_wagon is fixed.")

        self.current_length = self.total_length
        logger.info(
            f"MT {self.instance_id}: Stepped to length {self.current_length:.2f} for frame {frame_idx}. Total wagons: {len(self.wagons)}.")

    def draw(
            self,
            frame: np.ndarray,
            tubuli_mask: np.ndarray,
            cfg: SyntheticDataConfig,
            seed_mask: Optional[np.ndarray] = None,  # Changed | None to a standard Optional
    ) -> List[dict]:  # Changed list[dict] to List[dict] for consistency
        """Rasterizes each wagon using the flexible contrast model."""
        logger.debug(
            f"MT {self.instance_id}: Drawing microtubule for frame {self.frame_idx}. Total wagons: {len(self.wagons)}.")
        abs_angle = self.base_orientation
        abs_pos = self.base_point.astype(np.float32)  # Ensure float32 for consistent calculations
        gt_info = []

        if not self.wagons:
            logger.warning(f"MT {self.instance_id}: No wagons to draw. Skipping drawing.")
            return gt_info

        for idx, w in enumerate(self.wagons):
            if w.length <= 1e-6:  # Skip drawing negligible wagons
                logger.debug(
                    f"MT {self.instance_id}: Skipping drawing wagon {idx} due to negligible length ({w.length:.6f}).")
                continue

            abs_angle += w.angle
            dx = w.length * np.cos(abs_angle)
            dy = w.length * np.sin(abs_angle)
            new_pos = abs_pos + np.array([dx, dy], dtype=np.float32)

            sigma_x = cfg.sigma_x * (1 + np.random.normal(0, cfg.width_var_std))
            sigma_y = cfg.sigma_y * (1 + np.random.normal(0, cfg.width_var_std))

            # Ensure sigma values are non-negative
            sigma_x = max(0.0, sigma_x)
            sigma_y = max(0.0, sigma_y)

            logger.debug(
                f"MT {self.instance_id}, Wagon {idx}: Drawing from {abs_pos.tolist()} to {new_pos.tolist()}. Length: {w.length:.2f}, Angle: {math.degrees(w.angle):.2f} deg, Absolute Angle: {math.degrees(abs_angle):.2f} deg. Sigmas: ({sigma_x:.2f}, {sigma_y:.2f}).")

            # 1. Start with the base contrast for all channels.
            base_contrast = cfg.tubulus_contrast

            # 2. Apply tip brightness factor if applicable.
            is_tip_wagon = idx == len(self.wagons) - 1
            if self.state == "growing" and is_tip_wagon:
                base_contrast *= cfg.tip_brightness_factor
                logger.debug(
                    f"MT {self.instance_id}, Wagon {idx}: Applying tip brightness factor ({cfg.tip_brightness_factor:.2f}) as it's growing tip.")

            # 3. Create the RGB contrast tuple.
            r_contrast = base_contrast
            g_contrast = base_contrast
            b_contrast = base_contrast

            # 4. For seed segments, add the red channel boost.
            is_seed_wagon = idx == 0
            if is_seed_wagon:
                r_contrast += cfg.seed_red_channel_boost
                logger.debug(
                    f"MT {self.instance_id}, Wagon {idx}: Applying seed red channel boost ({cfg.seed_red_channel_boost:.2f}).")

            color_contrast_rgb = (r_contrast, g_contrast, b_contrast)
            logger.debug(f"MT {self.instance_id}, Wagon {idx}: Final color contrast (RGB): {color_contrast_rgb}.")

            # 5. Call the drawing function with the final calculated values.
            additional_mask_to_pass = (seed_mask if is_seed_wagon else None)

            try:
                draw_gaussian_line_rgb(
                    frame,
                    tubuli_mask,
                    abs_pos,
                    new_pos,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    color_contrast_rgb=color_contrast_rgb,
                    mask_idx=self.instance_id,
                    additional_mask=additional_mask_to_pass,
                )
                logger.debug(f"MT {self.instance_id}, Wagon {idx}: Successfully drawn.")
            except Exception as e:
                logger.error(
                    f"MT {self.instance_id}, Wagon {idx}: Failed to draw Gaussian line from {abs_pos.tolist()} to {new_pos.tolist()}: {e}",
                    exc_info=True)
                # Continue to next wagon if one fails, or re-raise if critical for overall image.

            # Ground truth generation
            gt_info.append(
                {
                    "frame_idx": self.frame_idx,  # Set current frame index
                    "wagon_index": idx,
                    "start": abs_pos.tolist(),
                    "end": new_pos.tolist(),
                    "angle": float(w.angle),
                    "length": float(w.length),
                    "instance_id": self.instance_id,
                }
            )
            abs_pos = new_pos.copy()

        logger.debug(
            f"MT {self.instance_id}: Finished drawing all wagons. Generated {len(gt_info)} ground truth segments.")
        return gt_info