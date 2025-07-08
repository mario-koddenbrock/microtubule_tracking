import pytest

from config.spots import SpotConfig
from config.tuning import TuningConfig
from config.synthetic_data import SyntheticDataConfig


# A simple mock object that simulates optuna.Trial's behavior
class MockOptunaTrial:
    def __init__(self, params):
        self.params = params
        self.number = params.get("number", 0)

    def suggest_float(self, name, low, high, *, log=False, step=None):
        return self.params.get(name)

    def suggest_int(self, name, low, high, step=1):
        return self.params.get(name)

    def suggest_categorical(self, name, choices):
        return self.params.get(name)


def test_create_config_from_trial():
    """
    Tests if TuningConfig can correctly create a SyntheticDataConfig
    using the new flexible rendering and bending parameters from a mocked Optuna trial.
    """
    tuning_cfg = TuningConfig()

    # Define a complete set of parameters matching the NEW create_synthetic_config_from_trial method.
    trial_params = {
        "number": 42,  # Example trial number

        # Kinematics
        "growth_speed": 2.0,
        "shrink_speed": 4.0,
        "catastrophe_prob": 0.01,
        "rescue_prob": 0.005,
        "pause_on_max_length": 5,
        "pause_on_min_length": 8,
        "min_length_min": 40, "min_length_max": 90,
        "max_length_min": 100, "max_length_max": 180,
        "bending_prob": 0.1,
        "um_per_pixel": 20,
        "scale_bar_um": 20,

        # Geometry & Shape
        "min_base_wagon_length": 15.0, "max_base_wagon_length": 45.0,
        "max_num_wagons": 10,
        "max_angle": 0.2,
        "max_angle_sign_changes": 1,
        "prob_to_flip_bend": 0.05,
        "min_wagon_length_min": 5, "min_wagon_length_max": 15,
        "max_wagon_length_min": 15, "max_wagon_length_max": 30,
        "num_tubuli": 25, "tubuli_seed_min_dist": 30, "margin": 10,

        # Rendering & Realism
        "sigma_x": 0.4, "sigma_y": 0.9,
        "tubule_width_variation": 0.1,
        "background_level": 0.7,
        "tubulus_contrast": -0.3,
        "seed_red_channel_boost": 0.6,
        "tip_brightness_factor": 1.2,
        "quantum_efficiency": 60.0,
        "gaussian_noise": 0.05,
        "bleach_tau": 500.0,
        "jitter_px": 0.4,
        "vignetting_strength": 0.1,
        "global_blur_sigma": 0.8,

        # Spots
        "fixed_spots_count": 10, "fixed_spots_intensity_min": 0.05, "fixed_spots_intensity_max": 0.1,
        "fixed_spots_radius_min": 1, "fixed_spots_radius_max": 2, "fixed_spots_kernel_size_min": 1,
        "fixed_spots_kernel_size_max": 2, "fixed_spots_sigma": 0.5,
        "moving_spots_count": 5, "moving_spots_intensity_min": 0.05, "moving_spots_intensity_max": 0.1,
        "moving_spots_radius_min": 1, "moving_spots_radius_max": 2, "moving_spots_kernel_size_min": 1,
        "moving_spots_kernel_size_max": 2, "moving_spots_sigma": 0.5, "moving_spots_max_step": 4,
        "random_spots_count": 5, "random_spots_intensity_min": 0.05, "random_spots_intensity_max": 0.1,
        "random_spots_radius_min": 1, "random_spots_radius_max": 2, "random_spots_kernel_size_min": 1,
        "random_spots_kernel_size_max": 2, "random_spots_sigma": 0.5,
    }
    mock_trial = MockOptunaTrial(trial_params)

    # Run the method we want to test
    synth_cfg = tuning_cfg.create_synthetic_config_from_trial(mock_trial)

    # Assert that the structure is correct and new values are populated
    assert isinstance(synth_cfg, SyntheticDataConfig)
    assert synth_cfg.tubulus_contrast == -0.3
    assert synth_cfg.seed_red_channel_boost == 0.6
    assert synth_cfg.max_angle_sign_changes == 1
    assert synth_cfg.prob_to_flip_bend == 0.05

    # Assert that nested objects are still created correctly
    assert isinstance(synth_cfg.fixed_spots, SpotConfig)
    assert synth_cfg.fixed_spots.count == 10
    assert synth_cfg.moving_spots.max_step == 4