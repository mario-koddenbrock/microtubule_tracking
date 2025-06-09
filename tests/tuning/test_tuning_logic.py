import pytest

from config.spots import SpotConfig
from config.tuning import TuningConfig


# A simple mock object that simulates optuna.Trial's behavior
class MockOptunaTrial:
    def __init__(self, params):
        self.params = params

    def suggest_float(self, name, low, high, *, log=False, step=None):
        return self.params[name]

    def suggest_int(self, name, low, high, step=1):
        return self.params[name]

    def suggest_categorical(self, name, choices):
        return self.params[name]


def test_create_config_from_trial():
    """
    Tests if TuningConfig can correctly create a nested SyntheticDataConfig
    from a mocked Optuna trial.
    """
    tuning_cfg = TuningConfig()

    # Define a set of parameters that a real trial might produce
    trial_params = {
        "shrink_frames": 15,
        "grow_frames": 25,
        "profile_noise": 5.0,
        "min_base_wagon_length": 20.0,
        "max_base_wagon_length": 30.0,
        "max_num_wagons": 5,
        "max_angle": 1.0,
        "max_angle_change_prob": 0.1,
        "min_wagon_length_min": 30,
        "min_wagon_length_max": 60,
        "max_wagon_length_min": 70,
        "max_wagon_length_max": 120,
        "min_length_min": 50,
        "min_length_max": 100,
        "max_length_min": 120,
        "max_length_max": 200,
        "num_tubulus": 20,
        "tubuli_min_dist": 10,
        "margin": 5,
        "pause_on_max_length": 2,
        "pause_on_min_length": 3,
        # CORRECTED: Add all missing parameters that your method suggests
        "width_var_std": 0.1,
        "bend_amplitude": 1.0,
        "bend_prob": 0.1,
        "bend_straight_fraction": 0.5,
        "sigma_x": 1.0,
        "sigma_y": 1.0,
        "background_level": 0.5,
        "tubulus_contrast": 0.2,
        "gaussian_noise": 0.1,
        "bleach_tau": 500.0,
        "jitter_px": 0.5,
        "vignetting_strength": 0.1,
        "invert_contrast": False,
        "global_blur_sigma": 0.5,
        "fixed_spots_count": 10,
        "fixed_spots_intensity_min": 0.1,
        "fixed_spots_intensity_max": 0.2,
        "fixed_spots_radius_min": 2,
        "fixed_spots_radius_max": 4,
        "fixed_spots_kernel_size_min": 1,
        "fixed_spots_kernel_size_max": 3,
        "fixed_spots_sigma": 1.5,
        "moving_spots_count": 0,  # Add all spot params, even if 0
        "moving_spots_intensity_min": 0.0,
        "moving_spots_intensity_max": 0.1,
        "moving_spots_radius_min": 1,
        "moving_spots_radius_max": 2,
        "moving_spots_kernel_size_min": 0,
        "moving_spots_kernel_size_max": 1,
        "moving_spots_sigma": 1.0,
        "moving_spots_max_step": 5,
        "random_spots_count": 0,
        "random_spots_intensity_min": 0.0,
        "random_spots_intensity_max": 0.1,
        "random_spots_radius_min": 1,
        "random_spots_radius_max": 2,
        "random_spots_kernel_size_min": 0,
        "random_spots_kernel_size_max": 1,
        "random_spots_sigma": 1.0,
        "show_time": False,
        "show_scale": False,
        "um_per_pixel": 0.1,
        "scale_bar_um": 5.0,
        "color_mode": False
    }

    mock_trial = MockOptunaTrial(trial_params)

    # Run the method we want to test
    synth_cfg = tuning_cfg.create_synthetic_config_from_trial(mock_trial)

    # Assert that the structure is correct
    assert synth_cfg.shrink_frames == 15
    assert isinstance(synth_cfg.fixed_spots, SpotConfig)
    assert synth_cfg.fixed_spots.count == 10
    assert synth_cfg.fixed_spots.sigma == 1.5