from dataclasses import dataclass
from typing import Optional, Tuple, List

from .base import BaseConfig


@dataclass(eq=False)
class TuningConfig(BaseConfig):
    """
    Configuration for hyperparameter tuning of synthetic microtubule data generation.

    Attributes:
        direction (str): Optimization direction ("maximize" or "minimize").
        metric (str): Metric to optimize (e.g., "cosine_similarity").
        model_name (str): Hugging Face model name for feature extraction.
        hf_cache_dir (str): Directory for Hugging Face model cache.
        reference_series_dir (str): Directory path with original reference video series.
        num_compare_series (int): Number of synthetic/reference series to compare.
        num_compare_frames (int): Number of frames to compare per series.
        temp_dir (str): Temporary directory for synthetic data generation.
        output_config_file (str): Where to store the optimal config JSON.
        num_trials (int): Number of optimization trials.

        # Motion profile parameters
        grow_frames (int): Number of frames over which growth occurs.
        shrink_frames (int): Number of frames over which shrinkage occurs.
        profile_noise (float): Noise level added to motion profile.
        pause_on_max_length (int): Frames to pause at maximum length.
        pause_on_min_length (int): Frames to pause at minimum length.

        # Tubule parameters
        motion (float): Scaling factor for motion per frame.
        min_length_min (int): Minimum lower bound for tubule length.
        min_length_max (int): Maximum lower bound for tubule length.
        max_length_min (int): Minimum upper bound for tubule length.
        max_length_max (int): Maximum upper bound for tubule length.
        num_tubulus_range (Tuple[int, int]): Range of number of tubules per series.
        tubuli_min_dist (int): Minimum distance between tubule seeds.
        margin (int): Border margin to keep tubules inside frame.

        # Bending and variation
        width_var_std (float): Std dev of width variation (relative).
        bend_amplitude (float): Maximum lateral offset for bending.
        bend_prob (float): Probability of a tubule being bent.
        bend_straight_fraction (float): Fraction of length before bending.

        # PSF / drawing width (ranges)
        sigma_x_range (Tuple[float, float]): Range for sigma_x.
        sigma_y_range (Tuple[float, float]): Range for sigma_y.

        # Photophysics / camera realism (ranges)
        background_level_range (Tuple[float, float]): Range for background intensity.
        tubulus_contrast_range (Tuple[float, float]): Range for tubule contrast.
        gaussian_noise_range (Tuple[float, float]): Range for Gaussian noise.
        bleach_tau_range (Tuple[float, float]): Range for bleaching time constant.
        jitter_px_range (Tuple[float, float]): Range for jitter in pixels.
        vignetting_strength_range (Tuple[float, float]): Range for vignetting strength.
        invert_contrast_options (List[bool]): Options for contrast inversion.
        global_blur_sigma_range (Tuple[float, float]): Range for global blur sigma.

        # Static spots (ranges)
        fixed_spot_count_range (Tuple[int, int]): Range for number of fixed spots.
        fixed_spot_intensity_range (Tuple[float, float]): Range for fixed spot intensity.
        fixed_spot_radius_range (Tuple[int, int]): Range for fixed spot radius.
        fixed_spot_kernel_size_range (Tuple[int, int]): Range for fixed spot kernel size.
        fixed_spot_sigma_range (Tuple[float, float]): Range for fixed spot sigma.

        # Moving spots (ranges)
        random_spot_count_range (Tuple[int, int]): Range for number of moving spots.
        random_spot_intensity_range (Tuple[float, float]): Range for moving spot intensity.
        random_spot_radius_range (Tuple[int, int]): Range for moving spot radius.
        random_spot_kernel_size_range (Tuple[int, int]): Range for moving spot kernel size.
        random_spot_sigma_range (Tuple[float, float]): Range for moving spot sigma.

        # Annotations
        show_time_options (List[bool]): Options to show or hide time.
        show_scale_options (List[bool]): Options to show or hide scale bar.
        um_per_pixel_range (Tuple[float, float]): Range for microns per pixel.
        scale_bar_um_range (Tuple[float, float]): Range for scale bar length in microns.
    """

    # Default values for “static” tuning parameters
    model_name: str = "openai/clip-vit-base-patch32"
    hf_cache_dir: Optional[str] = None
    reference_series_dir: str = "reference_data"
    num_compare_series: int = 3
    num_compare_frames: int = 1
    temp_dir: str = "temp_synthetic_data"
    output_config_file: str = "best_synthetic_config.json"
    output_config_id: int | str = "best_synthetic_config"
    output_config_num_frames: int = 30
    direction: str = "maximize"
    metric: str = "cosine_similarity"
    num_trials: int = 20

    # static video properties
    img_size: Tuple[int, int] = (462, 462)  # (H, W)
    fps: int = 5

    # ─── Motion-profile parameter ranges ───────────────────────────────────────
    grow_frames_range: Tuple[int, int] = (10, 30)
    shrink_frames_range: Tuple[int, int] = (5, 20)
    profile_noise_range: Tuple[float, float] = (0.0, 10.0)
    pause_on_max_length_range: Tuple[int, int] = (0, 10)
    pause_on_min_length_range: Tuple[int, int] = (0, 10)

    # ─── Tubule parameter ranges ───────────────────────────────────────────────
    motion_range: Tuple[float, float] = (0.5, 5.0)
    min_length_min_range: Tuple[int, int] = (20, 80)
    min_length_max_range: Tuple[int, int] = (min_length_min_range[1], 120)
    max_length_min_range: Tuple[int, int] = (80, 150)
    max_length_max_range: Tuple[int, int] = (max_length_min_range[1], 300)
    num_tubulus_range: Tuple[int, int] = (10, 30)
    tubuli_min_dist_range: Tuple[int, int] = (5, 50)
    margin_range: Tuple[int, int] = (0, 20)

    # ─── Bending & variation ranges ────────────────────────────────────────────
    width_var_std_range: Tuple[float, float] = (0.0, 0.2)
    bend_amplitude_range: Tuple[float, float] = (0.0, 10.0)
    bend_prob_range: Tuple[float, float] = (0.0, 0.5)
    bend_straight_fraction_range: Tuple[float, float] = (0.1, 1.0)

    # ─── PSF / drawing width ranges ────────────────────────────────────────────
    sigma_x_range: Tuple[float, float] = (0.1, 2.0)
    sigma_y_range: Tuple[float, float] = (0.1, 2.0)

    # ─── Photophysics / camera realism ranges ─────────────────────────────────
    background_level_range: Tuple[float, float] = (0.5, 1.0)
    tubulus_contrast_range: Tuple[float, float] = (0.0, 0.5)
    gaussian_noise_range: Tuple[float, float] = (0.0, 0.2)
    bleach_tau_range: Tuple[float, float] = (10.0, 1000.0)
    jitter_px_range: Tuple[float, float] = (0.0, 2.0)
    vignetting_strength_range: Tuple[float, float] = (0.0, 0.2)
    invert_contrast_options: List[bool] = (True, False)
    global_blur_sigma_range: Tuple[float, float] = (0.0, 3.0)

    # ─── Static spots ranges ───────────────────────────────────────────────────
    fixed_spot_count_range: Tuple[int, int] = (0, 100)
    fixed_spot_intensity_range: Tuple[float, float] = (0.0, 0.3)
    fixed_spot_radius_range: Tuple[int, int] = (1, 10)
    fixed_spot_kernel_size_range: Tuple[int, int] = (0, 10)
    fixed_spot_sigma_range: Tuple[float, float] = (0.1, 5.0)

    # ─── Moving spots ranges ───────────────────────────────────────────────────
    random_spot_count_range: Tuple[int, int] = (0, 50)
    random_spot_intensity_range: Tuple[float, float] = (0.0, 0.1)
    random_spot_radius_range: Tuple[int, int] = (1, 20)
    random_spot_kernel_size_range: Tuple[int, int] = (0, 10)
    random_spot_sigma_range: Tuple[float, float] = (0.1, 5.0)

    # ─── Annotation ranges ────────────────────────────────────────────────────
    show_time_options: List[bool] = (True, False)
    show_scale_options: List[bool] = (True, False)
    um_per_pixel_range: Tuple[float, float] = (0.01, 1.0)
    scale_bar_um_range: Tuple[float, float] = (1.0, 20.0)
    color_mode_options: List[bool] = (True, False)

    def validate(self):
        assert self.direction in ["maximize", "minimize"], "Direction must be 'maximize' or 'minimize'."
        assert self.num_trials > 0, "Number of trials must be positive."




