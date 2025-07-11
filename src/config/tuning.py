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
    Configuration for hyperparameter tuning of synthetic microtubule data generation,
    updated for the stateful, event-driven model.
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
    pca_components: Optional[int] = 0
    load_if_exists: bool = True

    # ─── Static Video Properties ────────────────────────────────────
    img_size: Tuple[int, int] = (462, 462)
    fps: int = 5

    # =========================================================================
    # PARAMETER RANGES FOR OPTIMIZATION
    # =========================================================================

    # ─── Microtubule kinematics ranges ───────────────────────
    growth_speed_range: Tuple[float, float] = (1.0, 5.0)
    shrink_speed_range: Tuple[float, float] = (5.0, 10.0)
    catastrophe_prob_range: Tuple[float, float] = (0.009, 0.011)
    rescue_prob_range: Tuple[float, float] = (0.009, 0.011)
    max_pause_at_min_frames_range: Tuple[int, int] = (0, 20)

    # ─── Microtubule geometry & bending ranges ────────────────────────
    base_wagon_length_min_range: Tuple[float, float] = (2.0, 20.0)
    base_wagon_length_max_range: Tuple[float, float] = (30.0, 80.0)
    microtubule_length_min_range: Tuple[int, int] = (50, 80)
    microtubule_length_max_range: Tuple[int, int] = (80, 200)
    tail_wagon_length_range: Tuple[float, float] = (5.0, 20.0)
    max_angle_range: Tuple[float, float] = (0.05, 0.4)
    max_angle_sign_changes_range: Tuple[int, int] = (0, 3)
    bending_prob_range: Tuple[float, float] = (0.001, 0.5)
    prob_to_flip_bend_range: Tuple[float, float] = (0.001, 0.5)

    # ─── Seeding & rendering ranges ───────────────────────
    num_microtubule_range: Tuple[int, int] = (5, 50)
    microtubule_seed_min_dist_range: Tuple[int, int] = (0, 50)
    margin_range: Tuple[int, int] = (0, 20)
    tubule_width_variation_range: Tuple[float, float] = (0.0, 0.2)
    psf_sigma_h_range: Tuple[float, float] = (0.2, 0.4)
    psf_sigma_v_range: Tuple[float, float] = (0.6, 0.9)

    # ─── Photophysics / camera realism ranges ─────────────────────
    background_level_range: Tuple[float, float] = (0.5, 0.9)
    tubulus_contrast_range: Tuple[float, float] = (-0.6, 0.6)
    seed_red_channel_boost_range: Tuple[float, float] = (0.0, 0.3)
    tip_brightness_factor_range: Tuple[float, float] = (1.0, 1.2)
    red_channel_noise_std_range: Tuple[float, float] = (0.0, 0.05)
    quantum_efficiency_range: Tuple[float, float] = (30.0, 70.0)
    gaussian_noise_range: Tuple[float, float] = (0.01, 0.1)
    bleach_tau_range: Tuple[float, float] = (100.0, 1000.0)
    jitter_px_range: Tuple[float, float] = (0.0, 1.0)
    vignetting_strength_range: Tuple[float, float] = (0.0, 0.2)
    global_blur_sigma_range: Tuple[float, float] = (0.3, 1.5)

    # ─── Spot Tuning Ranges ───────────────────────────────────────
    fixed_spots_tuning: SpotTuningConfig = field(default_factory=lambda: SpotTuningConfig(
        count_range=(0, 100), intensity_min_range=(0.0001, 0.1), intensity_max_range=(0.1, 0.3),
        polygon_p_range=(0.0, 0.7)
    ))
    moving_spots_tuning: SpotTuningConfig = field(default_factory=lambda: SpotTuningConfig(
        count_range=(0, 50), intensity_min_range=(0.0001, 0.1), intensity_max_range=(0.1, 0.3), max_step_range=(1, 10)
    ))
    random_spots_tuning: SpotTuningConfig = field(default_factory=lambda: SpotTuningConfig(
        count_range=(0, 50), intensity_min_range=(0.0001, 0.1), intensity_max_range=(0.0, 0.5)
    ))

    def __post_init__(self):
        super().__post_init__()
        logger.info("TuningConfig initialized. Running initial validation...")
        try:
            self.validate()
        except ValueError as e:
            logger.critical(f"Initial validation of TuningConfig failed: {e}", exc_info=False)
            raise

    def validate(self):
        """Validates all tuning configuration parameters."""
        logger.debug("Starting validation for TuningConfig...")
        errors = []

        def _validate_range(param_name: str, rng: Tuple[Union[int, float], Union[int, float]],
                            allow_zero_min: bool = True):
            if not isinstance(rng, (list, tuple)) or len(rng) != 2: errors.append(
                f"'{param_name}' must be a tuple of two numbers."); return
            min_val, max_val = rng
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)): errors.append(
                f"Elements of '{param_name}' must be numbers."); return
            if not allow_zero_min and min_val <= 0: errors.append(f"Min value of '{param_name}' must be positive.")
            if min_val > max_val: errors.append(f"Min value of '{param_name}' cannot be greater than max value.")

        # Validate all ranges
        range_params = [
            ("growth_speed_range", self.growth_speed_range, False),
            ("shrink_speed_range", self.shrink_speed_range, False),
            ("catastrophe_prob_range", self.catastrophe_prob_range, True),
            ("rescue_prob_range", self.rescue_prob_range, True),
            ("max_pause_at_min_frames_range", self.max_pause_at_min_frames_range, True),
            ("base_wagon_length_min_range", self.base_wagon_length_min_range, False),
            ("base_wagon_length_max_range", self.base_wagon_length_max_range, False),
            ("microtubule_length_min_range", self.microtubule_length_min_range, False),
            ("microtubule_length_max_range", self.microtubule_length_max_range, False),
            ("tail_wagon_length_range", self.tail_wagon_length_range, False),
            ("max_angle_range", self.max_angle_range, True),
            ("max_angle_sign_changes_range", self.max_angle_sign_changes_range, True),
            ("bending_prob_range", self.bending_prob_range, True),
            ("prob_to_flip_bend_range", self.prob_to_flip_bend_range, True),
            ("num_microtubule_range", self.num_microtubule_range, True),
            ("microtubule_seed_min_dist_range", self.microtubule_seed_min_dist_range, True),
            ("margin_range", self.margin_range, True),
            ("tubule_width_variation_range", self.tubule_width_variation_range, True),
            ("psf_sigma_h_range", self.psf_sigma_h_range, True), ("psf_sigma_v_range", self.psf_sigma_v_range, True),
            ("background_level_range", self.background_level_range, True),
            ("tubulus_contrast_range", self.tubulus_contrast_range, True),
            ("seed_red_channel_boost_range", self.seed_red_channel_boost_range, True),
            ("tip_brightness_factor_range", self.tip_brightness_factor_range, True),
            ("red_channel_noise_std_range", self.red_channel_noise_std_range, True),
            ("quantum_efficiency_range", self.quantum_efficiency_range, False),
            ("gaussian_noise_range", self.gaussian_noise_range, True),
            ("bleach_tau_range", self.bleach_tau_range, False),
            ("jitter_px_range", self.jitter_px_range, True),
            ("vignetting_strength_range", self.vignetting_strength_range, True),
            ("global_blur_sigma_range", self.global_blur_sigma_range, True),
        ]
        for name, rng, allow_zero in range_params: _validate_range(name, rng, allow_zero)

        if errors: raise ValueError(f"TuningConfig validation failed:\n" + "\n".join(errors))
        logger.info("TuningConfig validation successful.")

    def create_synthetic_config_from_trial(self, trial: Trial) -> SyntheticDataConfig:
        """
        Uses the ranges defined in this tuning config to suggest parameters for an Optuna trial.
        """
        logger.info(f"Generating SyntheticDataConfig for Optuna trial {trial.number}.")

        suggested_params = {}

        # --- Microtubule Kinematics ---
        suggested_params["growth_speed"] = trial.suggest_float("growth_speed", *self.growth_speed_range)
        suggested_params["shrink_speed"] = trial.suggest_float("shrink_speed", *self.shrink_speed_range)
        suggested_params["catastrophe_prob"] = trial.suggest_float("catastrophe_prob", *self.catastrophe_prob_range,
                                                                   log=True)
        suggested_params["rescue_prob"] = trial.suggest_float("rescue_prob", *self.rescue_prob_range, log=True)
        suggested_params["max_pause_at_min_frames"] = trial.suggest_int("max_pause_at_min_frames",
                                                                        *self.max_pause_at_min_frames_range)

        # --- Microtubule Geometry, Bending, Seeding, and Rendering ---
        suggested_params["base_wagon_length_min"] = trial.suggest_float("base_wagon_length_min",
                                                                        *self.base_wagon_length_min_range)
        suggested_params["base_wagon_length_max"] = trial.suggest_float("base_wagon_length_max",
                                                                        max(suggested_params["base_wagon_length_min"],
                                                                            self.base_wagon_length_max_range[0]),
                                                                        self.base_wagon_length_max_range[1])

        suggested_params["microtubule_length_min"] = trial.suggest_int("microtubule_length_min",
                                                                       max(suggested_params["base_wagon_length_max"],
                                                                           self.microtubule_length_min_range[0]),
                                                                       self.microtubule_length_min_range[1])
        suggested_params["microtubule_length_max"] = trial.suggest_int("microtubule_length_max",
                                                                       max(suggested_params["microtubule_length_min"],
                                                                           self.microtubule_length_max_range[0]),
                                                                       self.microtubule_length_max_range[1])

        suggested_params["tail_wagon_length"] = trial.suggest_float("tail_wagon_length", *self.tail_wagon_length_range)
        suggested_params["max_angle"] = trial.suggest_float("max_angle", *self.max_angle_range)
        suggested_params["max_angle_sign_changes"] = trial.suggest_int("max_angle_sign_changes",
                                                                       *self.max_angle_sign_changes_range)
        suggested_params["bending_prob"] = trial.suggest_float("bending_prob", *self.bending_prob_range)
        suggested_params["prob_to_flip_bend"] = trial.suggest_float("prob_to_flip_bend", *self.prob_to_flip_bend_range,
                                                                    log=True)

        suggested_params["num_microtubule"] = trial.suggest_int("num_microtubule", *self.num_microtubule_range)
        suggested_params["microtubule_seed_min_dist"] = trial.suggest_int("microtubule_seed_min_dist",
                                                                          *self.microtubule_seed_min_dist_range)
        suggested_params["margin"] = trial.suggest_int("margin", *self.margin_range)
        suggested_params["tubule_width_variation"] = trial.suggest_float("tubule_width_variation",
                                                                         *self.tubule_width_variation_range)
        suggested_params["psf_sigma_h"] = trial.suggest_float("psf_sigma_h", *self.psf_sigma_h_range)
        suggested_params["psf_sigma_v"] = trial.suggest_float("psf_sigma_v", *self.psf_sigma_v_range)

        # --- Photophysics & Camera Realism ---
        suggested_params["background_level"] = trial.suggest_float("background_level", *self.background_level_range)
        suggested_params["tubulus_contrast"] = trial.suggest_float("tubulus_contrast", *self.tubulus_contrast_range)
        suggested_params["seed_red_channel_boost"] = trial.suggest_float("seed_red_channel_boost",
                                                                         *self.seed_red_channel_boost_range)
        suggested_params["tip_brightness_factor"] = trial.suggest_float("tip_brightness_factor",
                                                                        *self.tip_brightness_factor_range)
        suggested_params["red_channel_noise_std"] = trial.suggest_float("red_channel_noise_std",
                                                                        *self.red_channel_noise_std_range)
        suggested_params["quantum_efficiency"] = trial.suggest_float("quantum_efficiency",
                                                                     *self.quantum_efficiency_range)
        suggested_params["gaussian_noise"] = trial.suggest_float("gaussian_noise", *self.gaussian_noise_range)
        suggested_params["bleach_tau"] = trial.suggest_float("bleach_tau", *self.bleach_tau_range, log=True)
        suggested_params["jitter_px"] = trial.suggest_float("jitter_px", *self.jitter_px_range)
        suggested_params["vignetting_strength"] = trial.suggest_float("vignetting_strength",
                                                                      *self.vignetting_strength_range)
        suggested_params["global_blur_sigma"] = trial.suggest_float("global_blur_sigma", *self.global_blur_sigma_range)

        # --- Spots ---
        fixed_spots_cfg = SpotConfig.from_trial(trial, "fixed_spots", self.fixed_spots_tuning)
        moving_spots_cfg = SpotConfig.from_trial(trial, "moving_spots", self.moving_spots_tuning)
        random_spots_cfg = SpotConfig.from_trial(trial, "random_spots", self.random_spots_tuning)

        # --- Build the final config object ---
        synth_cfg = SyntheticDataConfig(
            id=f"trial_{trial.number}",
            img_size=self.img_size,
            fps=self.fps,
            num_frames=self.num_compare_frames,
            fixed_spots=fixed_spots_cfg,
            moving_spots=moving_spots_cfg,
            random_spots=random_spots_cfg,
            **suggested_params
        )

        try:
            synth_cfg.validate()
        except ValueError as e:
            logger.warning(f"SyntheticDataConfig for trial {trial.number} failed validation: {e}. Pruning trial.")
            raise  # Re-raising will mark trial as failed/pruned by Optuna

        return synth_cfg