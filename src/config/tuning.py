from dataclasses import dataclass, field
from typing import Optional, Tuple

from optuna import Trial

from .base import BaseConfig
from .spots import SpotTuningConfig, SpotConfig
from .synthetic_data import SyntheticDataConfig


@dataclass(eq=False)
class TuningConfig(BaseConfig):
    """
    Configuration for hyperparameter tuning of synthetic microtubule data generation.
    """
    # ─── General tuning settings ────────────────────────────────────
    model_name: str = "openai/clip-vit-base-patch32"
    hf_cache_dir: Optional[str] = None
    reference_series_dir: str = "reference_data"
    num_compare_series: int = 3
    num_compare_frames: int = 1
    temp_dir: str = "temp_synthetic_data"
    output_config_file: str = "best_synthetic_config.json"
    output_config_id: int | str = "best_synthetic_config"
    output_config_num_frames: int = 50
    direction: str = "maximize"
    metric: str = "cosine_similarity"
    num_trials: int = 100
    pca_components: Optional[int] = 64
    load_if_exists: bool = False

    # ─── Static Video Properties ────────────────────────────────────
    img_size: Tuple[int, int] = (462, 462)
    fps: int = 5

    # =========================================================================
    # PARAMETER RANGES FOR OPTIMIZATION
    # =========================================================================

    # ─── Motion-profile (stochastic) ranges ───────────────────────
    growth_speed_range: Tuple[float, float] = (1.0, 5.0)
    shrink_speed_range: Tuple[float, float] = (2.0, 10.0)
    catastrophe_prob_range: Tuple[float, float] = (0.001, 0.05)
    rescue_prob_range: Tuple[float, float] = (0.0005, 0.02)
    pause_on_max_length_range: Tuple[int, int] = (0, 10)
    pause_on_min_length_range: Tuple[int, int] = (0, 10)

    # ─── Wagon & Bending kinematics ranges ────────────────────────
    min_base_wagon_length_range: Tuple[float, float] = (10.0, 50.0)
    max_base_wagon_length_range: Tuple[float, float] = (10.0, 50.0)
    max_num_wagons_range: Tuple[int, int] = (1, 20)
    max_angle_range: Tuple[float, float] = (0.05, 0.4)
    max_angle_sign_changes_range: Tuple[int, int] = (0, 3)
    bending_prob_range: Tuple[float, float] = (0.001, 0.5)
    prob_to_flip_bend_range: Tuple[float, float] = (0.001, 0.1)
    min_wagon_length_min_range: Tuple[int, int] = (1, 10)
    min_wagon_length_max_range: Tuple[int, int] = (10, 20)
    max_wagon_length_min_range: Tuple[int, int] = (10, 20)
    max_wagon_length_max_range: Tuple[int, int] = (20, 50)

    # ─── Tubule geometry & rendering ranges ───────────────────────
    min_length_min_range: Tuple[int, int] = (20, 80)
    min_length_max_range: Tuple[int, int] = (80, 120)
    max_length_min_range: Tuple[int, int] = (80, 150)
    max_length_max_range: Tuple[int, int] = (150, 300)
    num_tubulus_range: Tuple[int, int] = (10, 40)
    tubuli_seed_min_dist_range: Tuple[int, int] = (10, 50)
    margin_range: Tuple[int, int] = (0, 20)
    width_var_std_range: Tuple[float, float] = (0.0, 0.2)
    sigma_x_range: Tuple[float, float] = (0.1, 1.0)
    sigma_y_range: Tuple[float, float] = (0.1, 1.5)

    # ─── Photophysics / camera realism ranges ─────────────────────
    background_level_range: Tuple[float, float] = (0.5, 0.9)
    tubulus_contrast_range: Tuple[float, float] = (-0.6, 0.6)
    seed_red_channel_boost_range: Tuple[float, float] = (0.1, 1.0)
    tip_brightness_factor_range: Tuple[float, float] = (1.0, 2.0)
    quantum_efficiency_range: Tuple[float, float] = (20.0, 150.0)
    gaussian_noise_range: Tuple[float, float] = (0.01, 0.15)
    bleach_tau_range: Tuple[float, float] = (100.0, 1000.0)
    jitter_px_range: Tuple[float, float] = (0.0, 1.0)
    vignetting_strength_range: Tuple[float, float] = (0.0, 0.2)
    global_blur_sigma_range: Tuple[float, float] = (0.3, 1.5)

    # ─── Spot Tuning Ranges ───────────────────────────────────────
    fixed_spots_tuning: SpotTuningConfig = field(default_factory=lambda: SpotTuningConfig(
        count_range=(0, 100), intensity_min_range=(0.0001, 0.1), intensity_max_range=(0.1, 0.3)
    ))
    moving_spots_tuning: SpotTuningConfig = field(default_factory=lambda: SpotTuningConfig(
        count_range=(0, 50), intensity_min_range=(0.0001, 0.1), intensity_max_range=(0.1, 0.3), max_step_range=(1, 10)
    ))
    random_spots_tuning: SpotTuningConfig = field(default_factory=lambda: SpotTuningConfig(
        count_range=(0, 50), intensity_min_range=(0.0001, 0.1), intensity_max_range=(0.0, 0.5)
    ))

    # ─── Annotation ranges (typically not tuned, but included for completeness) ───
    show_time_options: Tuple[bool] = (True, False)
    show_scale_options: Tuple[bool] = (True, False)
    um_per_pixel_range: Tuple[float, float] = (0.05, 0.2)
    scale_bar_um_range: Tuple[float, float] = (2.0, 20.0)

    def validate(self):
        assert self.direction in ["maximize", "minimize"]
        assert self.num_trials > 0

    def create_synthetic_config_from_trial(self, trial: Trial) -> SyntheticDataConfig:
        """
        Uses the ranges defined in this tuning config to suggest parameters for
        an Optuna trial and returns a corresponding SyntheticDataConfig.
        """
        # --- Suggest all tunable parameters from the trial ---

        # Kinematics
        growth_speed = trial.suggest_float("growth_speed", *self.growth_speed_range)
        shrink_speed = trial.suggest_float("shrink_speed", *self.shrink_speed_range)
        catastrophe_prob = trial.suggest_float("catastrophe_prob", *self.catastrophe_prob_range, log=True)
        rescue_prob = trial.suggest_float("rescue_prob", *self.rescue_prob_range, log=True)
        pause_on_max_length = trial.suggest_int("pause_on_max_length", *self.pause_on_max_length_range)
        pause_on_min_length = trial.suggest_int("pause_on_min_length", *self.pause_on_min_length_range)
        min_length_min = trial.suggest_int("min_length_min", *self.min_length_min_range)
        min_length_max = trial.suggest_int("min_length_max", max(min_length_min + 1, self.min_length_max_range[0]),
                                           self.min_length_max_range[1])
        max_length_min = trial.suggest_int("max_length_min", *self.max_length_min_range)
        max_length_max = trial.suggest_int("max_length_max", max(max_length_min + 1, self.max_length_max_range[0]),
                                           self.max_length_max_range[1])

        # Geometry & Shape
        min_base_wagon_length = trial.suggest_float("min_base_wagon_length", *self.min_base_wagon_length_range)
        max_base_wagon_length = trial.suggest_float("max_base_wagon_length", *self.max_base_wagon_length_range)
        max_num_wagons = trial.suggest_int("max_num_wagons", *self.max_num_wagons_range)
        max_angle = trial.suggest_float("max_angle", *self.max_angle_range)
        max_angle_sign_changes = trial.suggest_int("max_angle_sign_changes", *self.max_angle_sign_changes_range)
        bending_prob = trial.suggest_float("bending_prob", *self.bending_prob_range)
        prob_to_flip_bend = trial.suggest_float("prob_to_flip_bend", *self.prob_to_flip_bend_range)
        min_wagon_length_min = trial.suggest_int("min_wagon_length_min", *self.min_wagon_length_min_range)
        min_wagon_length_max = trial.suggest_int("min_wagon_length_max",
                                                 max(min_wagon_length_min + 1, self.min_wagon_length_max_range[0]),
                                                 self.min_wagon_length_max_range[1])
        max_wagon_length_min = trial.suggest_int("max_wagon_length_min", *self.max_wagon_length_min_range)
        max_wagon_length_max = trial.suggest_int("max_wagon_length_max",
                                                 max(max_wagon_length_min + 1, self.max_wagon_length_max_range[0]),
                                                 self.max_wagon_length_max_range[1])
        num_tubulus = trial.suggest_int("num_tubulus", *self.num_tubulus_range)
        tubuli_seed_min_dist = trial.suggest_int("tubuli_seed_min_dist", *self.tubuli_seed_min_dist_range)
        margin = trial.suggest_int("margin", *self.margin_range)

        # Rendering & Realism
        sigma_x = trial.suggest_float("sigma_x", *self.sigma_x_range)
        sigma_y = trial.suggest_float("sigma_y", *self.sigma_y_range)
        width_var_std = trial.suggest_float("width_var_std", *self.width_var_std_range)
        background_level = trial.suggest_float("background_level", *self.background_level_range)
        tubulus_contrast = trial.suggest_float("tubulus_contrast", *self.tubulus_contrast_range)
        seed_red_channel_boost = trial.suggest_float("seed_red_channel_boost", *self.seed_red_channel_boost_range)
        tip_brightness_factor = trial.suggest_float("tip_brightness_factor", *self.tip_brightness_factor_range)
        quantum_efficiency = trial.suggest_float("quantum_efficiency", *self.quantum_efficiency_range)
        gaussian_noise = trial.suggest_float("gaussian_noise", *self.gaussian_noise_range)
        bleach_tau = trial.suggest_float("bleach_tau", *self.bleach_tau_range)
        jitter_px = trial.suggest_float("jitter_px", *self.jitter_px_range)
        vignetting_strength = trial.suggest_float("vignetting_strength", *self.vignetting_strength_range)
        global_blur_sigma = trial.suggest_float("global_blur_sigma", *self.global_blur_sigma_range)

        # Spots
        fixed_spots_cfg = SpotConfig.from_trial(trial, "fixed_spots", self.fixed_spots_tuning)
        moving_spots_cfg = SpotConfig.from_trial(trial, "moving_spots", self.moving_spots_tuning)
        random_spots_cfg = SpotConfig.from_trial(trial, "random_spots", self.random_spots_tuning)

        # Annotations
        show_time = trial.suggest_categorical("show_time", self.show_time_options)
        show_scale = trial.suggest_categorical("show_scale", self.show_scale_options)
        um_per_pixel = trial.suggest_float("um_per_pixel", *self.um_per_pixel_range)
        scale_bar_um = trial.suggest_float("scale_bar_um", *self.scale_bar_um_range)


        # Build the final config object
        synth_cfg = SyntheticDataConfig(
            id="current_trial",
            img_size=self.img_size,
            fps=self.fps,
            num_frames=self.num_compare_frames,
            color_mode=True,  # Always color for this model
            growth_speed=growth_speed,
            shrink_speed=shrink_speed,
            catastrophe_prob=catastrophe_prob,
            rescue_prob=rescue_prob,
            min_base_wagon_length=min_base_wagon_length,
            max_base_wagon_length=max_base_wagon_length,
            max_num_wagons=max_num_wagons,
            max_angle=max_angle,
            max_angle_sign_changes=max_angle_sign_changes,
            bending_prob=bending_prob,
            prob_to_flip_bend=prob_to_flip_bend,
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
            tubuli_seed_min_dist=tubuli_seed_min_dist,
            margin=margin,
            width_var_std=width_var_std,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            background_level=background_level,
            tubulus_contrast=tubulus_contrast,
            seed_red_channel_boost=seed_red_channel_boost,
            tip_brightness_factor=tip_brightness_factor,
            quantum_efficiency=quantum_efficiency,
            gaussian_noise=gaussian_noise,
            bleach_tau=bleach_tau,
            jitter_px=jitter_px,
            vignetting_strength=vignetting_strength,
            global_blur_sigma=global_blur_sigma,
            fixed_spots=fixed_spots_cfg,
            moving_spots=moving_spots_cfg,
            random_spots=random_spots_cfg,
            annotation_color_rgb=(1.0, 1.0, 1.0),
            show_time=show_time,
            show_scale=show_scale,
            um_per_pixel=um_per_pixel,
            scale_bar_um=scale_bar_um,
        )

        synth_cfg.validate()
        return synth_cfg