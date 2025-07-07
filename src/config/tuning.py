import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

from optuna import Trial

from .base import BaseConfig
from .spots import SpotTuningConfig, SpotConfig
from .synthetic_data import SyntheticDataConfig

logger = logging.getLogger(f"mt.{__name__}")


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
    similarity_metric: str = "mahalanobis"
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
    num_tubuli_range: Tuple[int, int] = (10, 40)
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
    show_time_options: Tuple[bool, ...] = (True, False)
    show_scale_options: Tuple[bool, ...] = (True, False)
    um_per_pixel_range: Tuple[float, float] = (0.05, 0.2)
    scale_bar_um_range: Tuple[float, float] = (2.0, 20.0)

    # ─── post-initialization and validation ─────────────────────
    def __post_init__(self):
        super().__post_init__()  # Call the base class's __post_init__
        logger.info("TuningConfig initialized. Running initial validation...")
        try:
            self.validate()
        except ValueError as e:
            logger.critical(f"Initial validation of TuningConfig failed: {e}", exc_info=False)
            raise  # Re-raise the error as it's a critical configuration issue.

    def validate(self):
        """
        Validates all tuning configuration parameters, including nested configs.
        Raises ValueError if any parameter is invalid.
        """
        logger.debug("Starting validation for TuningConfig...")
        errors = []

        # Helper to validate a range tuple
        def _validate_range(param_name: str, rng: Tuple[Union[int, float], Union[int, float]],
                            allow_zero_min: bool = False, type_str: str = "number"):
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                errors.append(f"'{param_name}' must be a tuple of two {type_str}s, but got {rng}.")
                return
            min_val, max_val = rng
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                errors.append(f"Elements of '{param_name}' must be {type_str}s, but got {rng}.")
                return
            if not allow_zero_min and min_val <= 0:
                errors.append(f"Min value of '{param_name}' must be positive, but got {min_val}.")
            if min_val > max_val:
                errors.append(f"Min value of '{param_name}' ({min_val}) cannot be greater than max value ({max_val}).")

        # --- General tuning settings ---
        if self.direction not in ["maximize", "minimize"]:
            errors.append(f"Direction must be 'maximize' or 'minimize', but got '{self.direction}'.")
        if not (self.num_trials > 0):
            errors.append(f"num_trials must be greater than 0, but got {self.num_trials}.")
        if not (self.num_compare_series > 0):
            errors.append(f"num_compare_series must be greater than 0, but got {self.num_compare_series}.")
        if not (self.num_compare_frames > 0):
            errors.append(f"num_compare_frames must be greater than 0, but got {self.num_compare_frames}.")
        if self.pca_components is not None and not (isinstance(self.pca_components, int) and self.pca_components > 0):
            errors.append(f"pca_components must be a positive integer or None, but got {self.pca_components}.")

        # --- Static Video Properties ---
        if not (isinstance(self.img_size, (list, tuple)) and len(self.img_size) == 2 and all(
                isinstance(x, int) and x > 0 for x in self.img_size)):
            errors.append(f"img_size must be a tuple of two positive integers, but got {self.img_size}.")
        if not (self.fps > 0):
            errors.append(f"fps must be greater than 0, but got {self.fps}.")

        # --- Validate all ranges using the helper ---
        range_params = {
            "growth_speed_range": (self.growth_speed_range, False, "float"),
            "shrink_speed_range": (self.shrink_speed_range, False, "float"),
            "catastrophe_prob_range": (self.catastrophe_prob_range, False, "float"),
            "rescue_prob_range": (self.rescue_prob_range, False, "float"),
            "pause_on_max_length_range": (self.pause_on_max_length_range, True, "int"),
            "pause_on_min_length_range": (self.pause_on_min_length_range, True, "int"),
            "min_base_wagon_length_range": (self.min_base_wagon_length_range, False, "float"),
            "max_base_wagon_length_range": (self.max_base_wagon_length_range, False, "float"),
            "max_num_wagons_range": (self.max_num_wagons_range, False, "int"),
            "max_angle_range": (self.max_angle_range, True, "float"),  # Max angle can be 0 (no bending)
            "max_angle_sign_changes_range": (self.max_angle_sign_changes_range, True, "int"),
            "bending_prob_range": (self.bending_prob_range, True, "float"),
            "prob_to_flip_bend_range": (self.prob_to_flip_bend_range, True, "float"),
            "min_wagon_length_min_range": (self.min_wagon_length_min_range, False, "int"),
            "min_wagon_length_max_range": (self.min_wagon_length_max_range, False, "int"),
            "max_wagon_length_min_range": (self.max_wagon_length_min_range, False, "int"),
            "max_wagon_length_max_range": (self.max_wagon_length_max_range, False, "int"),
            "min_length_min_range": (self.min_length_min_range, False, "int"),
            "min_length_max_range": (self.min_length_max_range, False, "int"),
            "max_length_min_range": (self.max_length_min_range, False, "int"),
            "max_length_max_range": (self.max_length_max_range, False, "int"),
            "num_tubuli_range": (self.num_tubuli_range, True, "int"),
            "tubuli_seed_min_dist_range": (self.tubuli_seed_min_dist_range, True, "int"),
            "margin_range": (self.margin_range, True, "int"),
            "width_var_std_range": (self.width_var_std_range, True, "float"),
            "sigma_x_range": (self.sigma_x_range, True, "float"),
            "sigma_y_range": (self.sigma_y_range, True, "float"),
            "background_level_range": (self.background_level_range, True, "float"),
            "tubulus_contrast_range": (self.tubulus_contrast_range, True, "float"),  # Can be negative
            "seed_red_channel_boost_range": (self.seed_red_channel_boost_range, True, "float"),
            "tip_brightness_factor_range": (self.tip_brightness_factor_range, True, "float"),
            "quantum_efficiency_range": (self.quantum_efficiency_range, False, "float"),
            "gaussian_noise_range": (self.gaussian_noise_range, True, "float"),
            "bleach_tau_range": (self.bleach_tau_range, False, "float"),  # Bleach tau must be > 0
            "jitter_px_range": (self.jitter_px_range, True, "float"),
            "vignetting_strength_range": (self.vignetting_strength_range, True, "float"),
            "global_blur_sigma_range": (self.global_blur_sigma_range, True, "float"),
            "um_per_pixel_range": (self.um_per_pixel_range, False, "float"),
            "scale_bar_um_range": (self.scale_bar_um_range, False, "float"),
        }

        for param_name, (rng, allow_zero, type_str) in range_params.items():
            _validate_range(param_name, rng, allow_zero, type_str)

        # Check specific logical constraints between ranges
        if self.min_length_max_range[0] > self.max_length_min_range[1]:
            errors.append(
                f"The lower bound of min_length_max_range ({self.min_length_max_range[0]}) is greater than the upper bound of max_length_min_range ({self.max_length_min_range[1]}). This could lead to impossible length suggestions.")
        if self.min_wagon_length_max_range[0] > self.max_wagon_length_min_range[1]:
            errors.append(
                f"The lower bound of min_wagon_length_max_range ({self.min_wagon_length_max_range[0]}) is greater than the upper bound of max_wagon_length_min_range ({self.max_wagon_length_min_range[1]}). This could lead to impossible length suggestions.")

        # --- Recursive Validation for Nested SpotTuningConfigs ---
        try:
            self.fixed_spots_tuning.validate()
        except ValueError as e:
            errors.append(f"Fixed spots tuning config validation failed: {e}")
            logger.error("Nested fixed_spots_tuning config invalid.")

        try:
            self.moving_spots_tuning.validate()
        except ValueError as e:
            errors.append(f"Moving spots tuning config validation failed: {e}")
            logger.error("Nested moving_spots_tuning config invalid.")

        try:
            self.random_spots_tuning.validate()
        except ValueError as e:
            errors.append(f"Random spots tuning config validation failed: {e}")
            logger.error("Nested random_spots_tuning config invalid.")

        # --- Annotation options check ---
        if not all(isinstance(opt, bool) for opt in self.show_time_options) or len(self.show_time_options) == 0:
            errors.append(f"show_time_options must be a non-empty tuple of booleans, but got {self.show_time_options}.")
        if not all(isinstance(opt, bool) for opt in self.show_scale_options) or len(self.show_scale_options) == 0:
            errors.append(
                f"show_scale_options must be a non-empty tuple of booleans, but got {self.show_scale_options}.")

        # --- Final Error Check ---
        if errors:
            full_msg = f"TuningConfig validation failed with {len(errors)} error(s):\n" + "\n".join(errors)
            logger.error(full_msg)
            raise ValueError(full_msg)

        logger.info("TuningConfig validation successful.")

    def create_synthetic_config_from_trial(self, trial: Trial) -> SyntheticDataConfig:
        """
        Uses the ranges defined in this tuning config to suggest parameters for
        an Optuna trial and returns a corresponding SyntheticDataConfig.
        """
        logger.info(f"Generating SyntheticDataConfig for Optuna trial {trial.number}.")
        logger.debug(f"Trial attributes: {trial.params}")  # Can log current trial's parameters if desired.

        # --- Suggest all tunable parameters from the trial ---
        suggested_params = {}

        # Kinematics
        suggested_params["growth_speed"] = trial.suggest_float("growth_speed", *self.growth_speed_range)
        suggested_params["shrink_speed"] = trial.suggest_float("shrink_speed", *self.shrink_speed_range)
        suggested_params["catastrophe_prob"] = trial.suggest_float("catastrophe_prob", *self.catastrophe_prob_range,
                                                                   log=True)
        suggested_params["rescue_prob"] = trial.suggest_float("rescue_prob", *self.rescue_prob_range, log=True)
        suggested_params["pause_on_max_length"] = trial.suggest_int("pause_on_max_length",
                                                                    *self.pause_on_max_length_range)
        suggested_params["pause_on_min_length"] = trial.suggest_int("pause_on_min_length",
                                                                    *self.pause_on_min_length_range)

        suggested_params["min_length_min"] = trial.suggest_int("min_length_min", *self.min_length_min_range)
        # Constrain min_length_max to be >= min_length_min
        suggested_params["min_length_max"] = trial.suggest_int(
            "min_length_max",
            max(suggested_params["min_length_min"] + 1, self.min_length_max_range[0]),
            # Ensure it's at least min_length_min + 1
            self.min_length_max_range[1]
        )
        suggested_params["max_length_min"] = trial.suggest_int("max_length_min", *self.max_length_min_range)
        # Constrain max_length_max to be >= max_length_min
        suggested_params["max_length_max"] = trial.suggest_int(
            "max_length_max",
            max(suggested_params["max_length_min"] + 1, self.max_length_max_range[0]),
            # Ensure it's at least max_length_min + 1
            self.max_length_max_range[1]
        )
        logger.debug(
            f"Suggested kinematics parameters: { {k: suggested_params[k] for k in ['growth_speed', 'shrink_speed', 'catastrophe_prob', 'rescue_prob', 'pause_on_max_length', 'pause_on_min_length', 'min_length_min', 'min_length_max', 'max_length_min', 'max_length_max']} }")

        # Geometry & Shape
        suggested_params["min_base_wagon_length"] = trial.suggest_float("min_base_wagon_length",
                                                                        *self.min_base_wagon_length_range)
        suggested_params["max_base_wagon_length"] = trial.suggest_float("max_base_wagon_length",
                                                                        *self.max_base_wagon_length_range)
        suggested_params["max_num_wagons"] = trial.suggest_int("max_num_wagons", *self.max_num_wagons_range)
        suggested_params["max_angle"] = trial.suggest_float("max_angle", *self.max_angle_range)
        suggested_params["max_angle_sign_changes"] = trial.suggest_int("max_angle_sign_changes",
                                                                       *self.max_angle_sign_changes_range)
        suggested_params["bending_prob"] = trial.suggest_float("bending_prob", *self.bending_prob_range)
        suggested_params["prob_to_flip_bend"] = trial.suggest_float("prob_to_flip_bend", *self.prob_to_flip_bend_range)

        suggested_params["min_wagon_length_min"] = trial.suggest_int("min_wagon_length_min",
                                                                     *self.min_wagon_length_min_range)
        # Constrain min_wagon_length_max to be >= min_wagon_length_min
        suggested_params["min_wagon_length_max"] = trial.suggest_int(
            "min_wagon_length_max",
            max(suggested_params["min_wagon_length_min"] + 1, self.min_wagon_length_max_range[0]),
            self.min_wagon_length_max_range[1]
        )
        suggested_params["max_wagon_length_min"] = trial.suggest_int("max_wagon_length_min",
                                                                     *self.max_wagon_length_min_range)
        # Constrain max_wagon_length_max to be >= max_wagon_length_min
        suggested_params["max_wagon_length_max"] = trial.suggest_int(
            "max_wagon_length_max",
            max(suggested_params["max_wagon_length_min"] + 1, self.max_wagon_length_max_range[0]),
            self.max_wagon_length_max_range[1]
        )
        suggested_params["num_tubuli"] = trial.suggest_int("num_tubuli", *self.num_tubuli_range)
        suggested_params["tubuli_seed_min_dist"] = trial.suggest_int("tubuli_seed_min_dist",
                                                                     *self.tubuli_seed_min_dist_range)
        suggested_params["margin"] = trial.suggest_int("margin", *self.margin_range)
        logger.debug(
            f"Suggested geometry parameters: { {k: suggested_params[k] for k in ['min_base_wagon_length', 'max_base_wagon_length', 'max_num_wagons', 'max_angle', 'max_angle_sign_changes', 'bending_prob', 'prob_to_flip_bend', 'min_wagon_length_min', 'min_wagon_length_max', 'max_wagon_length_min', 'max_wagon_length_max', 'num_tubuli', 'tubuli_seed_min_dist', 'margin']} }")

        # Rendering & Realism
        suggested_params["sigma_x"] = trial.suggest_float("sigma_x", *self.sigma_x_range)
        suggested_params["sigma_y"] = trial.suggest_float("sigma_y", *self.sigma_y_range)
        suggested_params["width_var_std"] = trial.suggest_float("width_var_std", *self.width_var_std_range)
        suggested_params["background_level"] = trial.suggest_float("background_level", *self.background_level_range)
        suggested_params["tubulus_contrast"] = trial.suggest_float("tubulus_contrast", *self.tubulus_contrast_range)
        suggested_params["seed_red_channel_boost"] = trial.suggest_float("seed_red_channel_boost",
                                                                         *self.seed_red_channel_boost_range)
        suggested_params["tip_brightness_factor"] = trial.suggest_float("tip_brightness_factor",
                                                                        *self.tip_brightness_factor_range)
        suggested_params["quantum_efficiency"] = trial.suggest_float("quantum_efficiency",
                                                                     *self.quantum_efficiency_range)
        suggested_params["gaussian_noise"] = trial.suggest_float("gaussian_noise", *self.gaussian_noise_range)
        suggested_params["bleach_tau"] = trial.suggest_float("bleach_tau", *self.bleach_tau_range)
        suggested_params["jitter_px"] = trial.suggest_float("jitter_px", *self.jitter_px_range)
        suggested_params["vignetting_strength"] = trial.suggest_float("vignetting_strength",
                                                                      *self.vignetting_strength_range)
        suggested_params["global_blur_sigma"] = trial.suggest_float("global_blur_sigma", *self.global_blur_sigma_range)
        logger.debug(
            f"Suggested rendering/realism parameters: { {k: suggested_params[k] for k in ['sigma_x', 'sigma_y', 'width_var_std', 'background_level', 'tubulus_contrast', 'seed_red_channel_boost', 'tip_brightness_factor', 'quantum_efficiency', 'gaussian_noise', 'bleach_tau', 'jitter_px', 'vignetting_strength', 'global_blur_sigma']} }")

        # Spots - delegated to SpotConfig.from_trial
        logger.debug(f"Suggesting fixed_spots config using tuning: {self.fixed_spots_tuning.asdict()}")
        fixed_spots_cfg = SpotConfig.from_trial(trial, "fixed_spots", self.fixed_spots_tuning)

        logger.debug(f"Suggesting moving_spots config using tuning: {self.moving_spots_tuning.asdict()}")
        moving_spots_cfg = SpotConfig.from_trial(trial, "moving_spots", self.moving_spots_tuning)

        logger.debug(f"Suggesting random_spots config using tuning: {self.random_spots_tuning.asdict()}")
        random_spots_cfg = SpotConfig.from_trial(trial, "random_spots", self.random_spots_tuning)

        # Annotations
        suggested_params["show_time"] = trial.suggest_categorical("show_time", self.show_time_options)
        suggested_params["show_scale"] = trial.suggest_categorical("show_scale", self.show_scale_options)
        suggested_params["um_per_pixel"] = trial.suggest_float("um_per_pixel", *self.um_per_pixel_range)
        suggested_params["scale_bar_um"] = trial.suggest_float("scale_bar_um", *self.scale_bar_um_range)
        logger.debug(
            f"Suggested annotation parameters: { {k: suggested_params[k] for k in ['show_time', 'show_scale', 'um_per_pixel', 'scale_bar_um']} }")

        # Build the final config object
        synth_cfg = SyntheticDataConfig(
            id=f"trial_{trial.number}",  # Use trial ID for better traceability
            img_size=self.img_size,
            fps=self.fps,
            num_frames=self.num_compare_frames,
            color_mode=True,  # Fixed for this model
            fixed_spots=fixed_spots_cfg,
            moving_spots=moving_spots_cfg,
            random_spots=random_spots_cfg,
            annotation_color_rgb=(1.0, 1.0, 1.0),  # Fixed for this model TODO
            albumentations=None,  # Or AlbumentationsConfig() if you want to tune augmentations too
            generate_tubuli_mask=False,  # Fixed for this model
            generate_seed_mask=False,  # Fixed for this model
            **suggested_params  # Unpack all the dynamically suggested parameters
        )

        logger.debug(f"Attempting to validate generated SyntheticDataConfig for trial {trial.number}.")
        try:
            synth_cfg.validate()
            logger.info(f"SyntheticDataConfig for trial {trial.number} validated successfully.")
        except ValueError as e:
            logger.warning(
                f"SyntheticDataConfig for trial {trial.number} failed validation: {e}. This trial might be invalid.")
            # Note: Optuna can handle invalid trials by returning a large objective value or raising TrialPruned.
            # You might want to re-raise if it's a critical error that should stop the study.
            # For now, we just warn and proceed.
            raise  # Re-raising will mark trial as failed/pruned by Optuna

        return synth_cfg