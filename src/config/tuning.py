import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from optuna import Trial

from .base import BaseConfig
from .spots import SpotTuningConfig, SpotConfig
from .synthetic_data import SyntheticDataConfig


@dataclass(eq=False)
class TuningConfig(BaseConfig):
    """
    Configuration for hyperparameter tuning of synthetic microtubule data generation.
    (Cleaned and synchronized with SyntheticDataConfig)

    Attributes:
        # General tuning settings
        direction (str): Optimization direction ("maximize" or "minimize").
        model_name (str): Hugging Face model for feature extraction.
        ... (and other static settings) ...

        # Parameter Ranges for Optimization
        # Each '..._range' or '..._options' attribute defines the search space
        # for a corresponding parameter in SyntheticDataConfig.
        # The '..._spots_tuning' attributes contain nested tuning configurations
        # for fixed, moving, and random spots.
    """

    # --- Static Tuning Parameters ---
    model_name: str = "openai/clip-vit-base-patch32"
    hf_cache_dir: Optional[str] = None
    reference_series_dir: str = "reference_data"
    num_compare_series: int = 3
    num_compare_frames: int = 20  # Number of frames to generate per trial
    temp_dir: str = "temp_synthetic_data"
    output_config_file: str = "best_synthetic_config.json"
    output_config_id: int | str = "best_synthetic_config"
    output_config_num_frames: int = 30  # Number of frames for the final video
    direction: str = "maximize"
    metric: str = "cosine_similarity"
    num_trials: int = 100
    pca_components: Optional[int] = 64
    load_if_exists:bool = False

    # --- Static Video Properties ---
    img_size: Tuple[int, int] = (462, 462)
    fps: int = 5

    # =========================================================================
    # PARAMETER RANGES FOR OPTIMIZATION
    # =========================================================================

    # ─── Motion-profile ranges ───────────────────────────────────────
    grow_frames_range: Tuple[int, int] = (10, 30)
    shrink_frames_range: Tuple[int, int] = (5, 20)
    profile_noise_range: Tuple[float, float] = (0.0, 10.0)
    pause_on_max_length_range: Tuple[int, int] = (0, 10)
    pause_on_min_length_range: Tuple[int, int] = (0, 10)

    # ─── Wagon kinematics ranges (NEWLY ADDED) ────────────────────────
    mmin_base_wagon_length_range: Tuple[float, float] = (10.0, 50.0)
    max_base_wagon_length_range: Tuple[float, float] = (10.0, 50.0)
    max_num_wagons_range: Tuple[int, int] = (1, 10)
    max_angle_range: Tuple[float, float] = (0.1, math.pi / 2)
    max_angle_change_prob_range: Tuple[float, float] = (0.01, 0.2)
    min_wagon_length_min_range: Tuple[int, int] = (20, 80)
    min_wagon_length_max_range: Tuple[int, int] = (80, 120)
    max_wagon_length_min_range: Tuple[int, int] = (80, 150)
    max_wagon_length_max_range: Tuple[int, int] = (150, 300)

    # ─── Tubule geometry ranges (motion_range removed) ────────────────
    min_length_min_range: Tuple[int, int] = (20, 80)
    min_length_max_range: Tuple[int, int] = (80, 120)
    max_length_min_range: Tuple[int, int] = (80, 150)
    max_length_max_range: Tuple[int, int] = (150, 300)
    num_tubulus_range: Tuple[int, int] = (10, 30)
    tubuli_min_dist_range: Tuple[int, int] = (5, 50)
    margin_range: Tuple[int, int] = (0, 20)

    # ... (Bending, PSF, Photophysics ranges are the same) ...
    width_var_std_range: Tuple[float, float] = (0.0, 0.2)
    bend_amplitude_range: Tuple[float, float] = (0.0, 10.0)
    bend_prob_range: Tuple[float, float] = (0.0, 0.5)
    bend_straight_fraction_range: Tuple[float, float] = (0.1, 1.0)
    sigma_x_range: Tuple[float, float] = (0.1, 2.0)
    sigma_y_range: Tuple[float, float] = (0.1, 2.0)
    background_level_range: Tuple[float, float] = (0.5, 1.0)
    tubulus_contrast_range: Tuple[float, float] = (0.0, 0.5)
    gaussian_noise_range: Tuple[float, float] = (0.0, 0.2)
    bleach_tau_range: Tuple[float, float] = (10.0, 1000.0)
    jitter_px_range: Tuple[float, float] = (0.0, 2.0)
    vignetting_strength_range: Tuple[float, float] = (0.0, 0.2)
    invert_contrast_options: Tuple[bool] = (True, False)
    global_blur_sigma_range: Tuple[float, float] = (0.0, 3.0)

    # ─── Spot Tuning Ranges (Refactored) ──────────────────────────────────
    fixed_spots_tuning: SpotTuningConfig = field(default_factory=lambda: SpotTuningConfig(
        count_range=(0, 100),
        intensity_min_range=(0.0001, 0.1), # Range for the minimum value
        intensity_max_range=(0.1, 0.3)  # Range for the maximum value
    ))
    moving_spots_tuning: SpotTuningConfig = field(default_factory=lambda: SpotTuningConfig(
        count_range=(0, 50),
        intensity_min_range=(0.0001, 0.1),
        intensity_max_range=(0.1, 0.3),
        max_step_range=(1, 10)
    ))
    random_spots_tuning: SpotTuningConfig = field(default_factory=lambda: SpotTuningConfig(
        count_range=(0, 50),
        intensity_min_range=(0.0001, 0.1),
        intensity_max_range=(0.0, 0.5)
    ))

    # ─── Annotation ranges ────────────────────────────────────────────────
    show_time_options: Tuple[bool] = (True, False)
    show_scale_options: Tuple[bool] = (True, False)
    um_per_pixel_range: Tuple[float, float] = (0.01, 1.0)
    scale_bar_um_range: Tuple[float, float] = (1.0, 20.0)
    color_mode_options: Tuple[bool] = (True, False)

    def validate(self):
        assert self.direction in ["maximize", "minimize"], "Direction must be 'maximize' or 'minimize'."
        assert self.num_trials > 0, "Number of trials must be positive."

    def create_synthetic_config_from_trial(self, trial: Trial) -> SyntheticDataConfig:
        """
        Uses the ranges defined in this tuning config to suggest parameters for
        an Optuna trial and returns a corresponding SyntheticDataConfig.
        """
        # --- Suggest all parameters from trial ---

        # Motion profile
        shrink_frames = trial.suggest_int("shrink_frames", *self.shrink_frames_range)
        grow_frames = trial.suggest_int("grow_frames", max(shrink_frames + 1, self.grow_frames_range[0]),
                                        self.grow_frames_range[1])
        profile_noise = trial.suggest_float("profile_noise", *self.profile_noise_range)
        pause_on_max_length = trial.suggest_int("pause_on_max_length", *self.pause_on_max_length_range)
        pause_on_min_length = trial.suggest_int("pause_on_min_length", *self.pause_on_min_length_range)

        # Wagon kinematics (NEWLY ADDED)
        mmin_base_wagon_length = trial.suggest_float("mmin_base_wagon_length", *self.mmin_base_wagon_length_range)
        max_base_wagon_length = trial.suggest_float("max_base_wagon_length", *self.max_base_wagon_length_range)
        max_num_wagons = trial.suggest_int("max_num_wagons", *self.max_num_wagons_range)
        max_angle = trial.suggest_float("max_angle", *self.max_angle_range)
        max_angle_change_prob = trial.suggest_float("max_angle_change_prob", *self.max_angle_change_prob_range)
        min_wagon_length_min = trial.suggest_int("min_wagon_length_min", *self.min_wagon_length_min_range)
        min_wagon_length_max = trial.suggest_int("min_wagon_length_max",
                                                 max(min_wagon_length_min + 1, self.min_wagon_length_max_range[0]),
                                                 self.min_wagon_length_max_range[1])
        max_wagon_length_min = trial.suggest_int("max_wagon_length_min", *self.max_wagon_length_min_range)
        max_wagon_length_max = trial.suggest_int("max_wagon_length_max",
                                                 max(max_wagon_length_min + 1, self.max_wagon_length_max_range[0]),
                                                 self.max_wagon_length_max_range[1])

        # Tubule geometry (motion removed)
        min_length_min = trial.suggest_int("min_length_min", *self.min_length_min_range)
        min_length_max = trial.suggest_int("min_length_max", max(min_length_min + 1, self.min_length_max_range[0]),
                                           self.min_length_max_range[1])
        max_length_min = trial.suggest_int("max_length_min", *self.max_length_min_range)
        max_length_max = trial.suggest_int("max_length_max", max(max_length_min + 1, self.max_length_max_range[0]),
                                           self.max_length_max_range[1])
        num_tubulus = trial.suggest_int("num_tubulus", *self.num_tubulus_range)
        tubuli_min_dist = trial.suggest_int("tubuli_min_dist", *self.tubuli_min_dist_range)
        margin = trial.suggest_int("margin", *self.margin_range)

        width_var_std = trial.suggest_float("width_var_std", *self.width_var_std_range)
        global_blur_sigma = trial.suggest_float("global_blur_sigma", *self.global_blur_sigma_range)

        # Spots (Refactored)
        fixed_spots_cfg = SpotConfig.from_trial(trial, "fixed_spots", self.fixed_spots_tuning)
        moving_spots_cfg = SpotConfig.from_trial(trial, "moving_spots", self.moving_spots_tuning)
        random_spots_cfg = SpotConfig.from_trial(trial, "random_spots", self.random_spots_tuning)

        # Annotations
        show_time = trial.suggest_categorical("show_time", self.show_time_options)
        show_scale = trial.suggest_categorical("show_scale", self.show_scale_options)
        um_per_pixel = trial.suggest_float("um_per_pixel", *self.um_per_pixel_range)
        scale_bar_um = trial.suggest_float("scale_bar_um", *self.scale_bar_um_range)
        color_mode = trial.suggest_categorical("color_mode", self.color_mode_options)

        # --- Build the SyntheticDataConfig object ---
        synth_cfg = SyntheticDataConfig(
            # Static fields
            id="current_trial",
            img_size=self.img_size,
            fps=self.fps,
            num_frames=self.num_compare_frames,
            color_mode=color_mode,

            # Motion & Tubules
            grow_frames=grow_frames,
            shrink_frames=shrink_frames,
            profile_noise=profile_noise,
            mmin_base_wagon_length=mmin_base_wagon_length,
            max_base_wagon_length=max_base_wagon_length,
            max_num_wagons=max_num_wagons,
            max_angle=max_angle,
            max_angle_change_prob=max_angle_change_prob,
            min_length_min=min_length_min,
            min_length_max=min_length_max,
            max_length_min=max_length_min,
            max_length_max=max_length_max,
            min_wagon_length_min=min_wagon_length_min,
            min_wagon_length_max=min_wagon_length_max,
            max_wagon_length_min=max_wagon_length_min,
            max_wagon_length_max=max_wagon_length_max,
            pause_on_max_length=pause_on_max_length,
            pause_on_min_length=pause_on_min_length,
            num_tubulus=num_tubulus,
            tubuli_min_dist=tubuli_min_dist,
            margin=margin,
            width_var_std=width_var_std,
            global_blur_sigma=global_blur_sigma,

            # Spots
            fixed_spots=fixed_spots_cfg,
            moving_spots=moving_spots_cfg,
            random_spots=random_spots_cfg,

            # Annotations
            show_time=show_time,
            show_scale=show_scale,
            um_per_pixel=um_per_pixel,
            scale_bar_um=scale_bar_um,
        )

        synth_cfg.validate()
        return synth_cfg