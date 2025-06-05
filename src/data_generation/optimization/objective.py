import optuna

from config.tuning import TuningConfig
from config.synthetic_data import SyntheticDataConfig
from data_generation.optimization.run import run


def objective(trial: optuna.trial.Trial,
              tuning_cfg: TuningConfig,
              ref_embs,
              model,
              extractor) -> float:
    # ─── Motion‐profile parameters ───────────────────────────────────
    shrink_frames = trial.suggest_int(
        "shrink_frames", *tuning_cfg.shrink_frames_range
    )
    grow_frames = trial.suggest_int(
        "grow_frames",
        max(shrink_frames + 1, tuning_cfg.grow_frames_range[0]),
        tuning_cfg.grow_frames_range[1],
    )
    profile_noise = trial.suggest_float(
        "profile_noise", *tuning_cfg.profile_noise_range
    )
    pause_on_max_length = trial.suggest_int(
        "pause_on_max_length", *tuning_cfg.pause_on_max_length_range
    )
    pause_on_min_length = trial.suggest_int(
        "pause_on_min_length", *tuning_cfg.pause_on_min_length_range
    )

    # ─── Tubule geometry ─────────────────────────────────────────────
    motion = trial.suggest_float(
        "motion", *tuning_cfg.motion_range
    )

    min_length_min = trial.suggest_int(
        "min_length_min", *tuning_cfg.min_length_min_range
    )
    # ensure min_length_max > min_length_min
    min_length_max = trial.suggest_int(
        "min_length_max",
        max(min_length_min + 1, tuning_cfg.min_length_max_range[0]),
        tuning_cfg.min_length_max_range[1]
    )

    max_length_min = trial.suggest_int(
        "max_length_min", *tuning_cfg.max_length_min_range
    )
    # ensure max_length_max > max_length_min
    max_length_max = trial.suggest_int(
        "max_length_max",
        max(max_length_min + 1, tuning_cfg.max_length_max_range[0]),
        tuning_cfg.max_length_max_range[1]
    )

    num_tubulus = trial.suggest_int(
        "num_tubulus", *tuning_cfg.num_tubulus_range
    )
    tubuli_min_dist = trial.suggest_int(
        "tubuli_min_dist", *tuning_cfg.tubuli_min_dist_range
    )
    margin = trial.suggest_int(
        "margin", *tuning_cfg.margin_range
    )

    # ─── Bending & variation ────────────────────────────────────────
    width_var_std = trial.suggest_float(
        "width_var_std", *tuning_cfg.width_var_std_range
    )
    bend_amplitude = trial.suggest_float(
        "bend_amplitude", *tuning_cfg.bend_amplitude_range
    )
    bend_prob = trial.suggest_float(
        "bend_prob", *tuning_cfg.bend_prob_range
    )
    bend_straight_fraction = trial.suggest_float(
        "bend_straight_fraction",
        *tuning_cfg.bend_straight_fraction_range
    )

    # ─── PSF / drawing width ────────────────────────────────────────
    sigma_x = trial.suggest_float(
        "sigma_x", *tuning_cfg.sigma_x_range
    )
    sigma_y = trial.suggest_float(
        "sigma_y", *tuning_cfg.sigma_y_range
    )

    # ─── Photophysics / camera realism ─────────────────────────────
    background_level = trial.suggest_float(
        "background_level", *tuning_cfg.background_level_range
    )
    tubulus_contrast = trial.suggest_float(
        "tubulus_contrast", *tuning_cfg.tubulus_contrast_range
    )
    gaussian_noise = trial.suggest_float(
        "gaussian_noise", *tuning_cfg.gaussian_noise_range
    )
    bleach_tau = trial.suggest_float(
        "bleach_tau", *tuning_cfg.bleach_tau_range
    )
    jitter_px = trial.suggest_float(
        "jitter_px", *tuning_cfg.jitter_px_range
    )
    vignetting_strength = trial.suggest_float(
        "vignetting_strength", *tuning_cfg.vignetting_strength_range
    )
    invert_contrast = trial.suggest_categorical(
        "invert_contrast", tuning_cfg.invert_contrast_options
    )
    global_blur_sigma = trial.suggest_float(
        "global_blur_sigma", *tuning_cfg.global_blur_sigma_range
    )

    # ─── Static spots ───────────────────────────────────────────────
    fixed_spot_count = trial.suggest_int(
        "fixed_spot_count", *tuning_cfg.fixed_spot_count_range
    )
    fixed_spot_intensity_min = trial.suggest_float(
        "fixed_spot_intensity_min", *tuning_cfg.fixed_spot_intensity_range
    )
    fixed_spot_intensity_max = trial.suggest_float(
        "fixed_spot_intensity_max",
        max(fixed_spot_intensity_min, tuning_cfg.fixed_spot_intensity_range[0]),
        tuning_cfg.fixed_spot_intensity_range[1],
    )
    fixed_spot_radius_min = trial.suggest_int(
        "fixed_spot_radius_min", *tuning_cfg.fixed_spot_radius_range
    )
    fixed_spot_radius_max = trial.suggest_int(
        "fixed_spot_radius_max",
        max(fixed_spot_radius_min, tuning_cfg.fixed_spot_radius_range[0]),
        tuning_cfg.fixed_spot_radius_range[1],
    )
    fixed_spot_kernel_size_min = trial.suggest_int(
        "fixed_spot_kernel_size_min", *tuning_cfg.fixed_spot_kernel_size_range
    )
    fixed_spot_kernel_size_max = trial.suggest_int(
        "fixed_spot_kernel_size_max",
        max(fixed_spot_kernel_size_min, tuning_cfg.fixed_spot_kernel_size_range[0]),
        tuning_cfg.fixed_spot_kernel_size_range[1],
    )
    fixed_spot_sigma = trial.suggest_float(
        "fixed_spot_sigma", *tuning_cfg.fixed_spot_sigma_range
    )

    # ─── Moving spots ───────────────────────────────────────────────
    random_spot_count = trial.suggest_int(
        "random_spot_count", *tuning_cfg.random_spot_count_range
    )
    random_spot_intensity_min = trial.suggest_float(
        "random_spot_intensity_min", *tuning_cfg.random_spot_intensity_range
    )
    random_spot_intensity_max = trial.suggest_float(
        "random_spot_intensity_max",
        max(random_spot_intensity_min, tuning_cfg.random_spot_intensity_range[0]),
        tuning_cfg.random_spot_intensity_range[1]
    )
    random_spot_radius_min = trial.suggest_int(
        "random_spot_radius_min", *tuning_cfg.random_spot_radius_range
    )
    random_spot_radius_max = trial.suggest_int(
        "random_spot_radius_max",
        max(random_spot_radius_min, tuning_cfg.random_spot_radius_range[0]),
        tuning_cfg.random_spot_radius_range[1],
    )
    random_spot_kernel_size_min = trial.suggest_int(
        "random_spot_kernel_size_min", *tuning_cfg.random_spot_kernel_size_range
    )
    random_spot_kernel_size_max = trial.suggest_int(
        "random_spot_kernel_size_max",
        max(random_spot_kernel_size_min, tuning_cfg.random_spot_kernel_size_range[0]),
        tuning_cfg.random_spot_kernel_size_range[1],
    )
    random_spot_sigma = trial.suggest_float(
        "random_spot_sigma", *tuning_cfg.random_spot_sigma_range
    )

    # ─── Annotations ────────────────────────────────────────────────
    show_time = trial.suggest_categorical(
        "show_time", tuning_cfg.show_time_options
    )
    show_scale = trial.suggest_categorical(
        "show_scale", tuning_cfg.show_scale_options
    )
    um_per_pixel = trial.suggest_float(
        "um_per_pixel", *tuning_cfg.um_per_pixel_range
    )
    scale_bar_um = trial.suggest_float(
        "scale_bar_um", *tuning_cfg.scale_bar_um_range
    )
    color_mode = trial.suggest_categorical(
        "show_time", tuning_cfg.color_mode_options
    )

    # Build a new SyntheticDataConfig object
    synth_cfg = SyntheticDataConfig(

        # carry over any static fields from tuning_cfg:
        generate_mask=False,
        id=0,
        img_size=tuning_cfg.img_size,
        fps=tuning_cfg.fps,
        num_frames=tuning_cfg.num_compare_frames,
        color_mode=color_mode,

        grow_frames=grow_frames,
        shrink_frames=shrink_frames,
        profile_noise=profile_noise,

        motion=motion,
        min_length_min=min_length_min,
        min_length_max=min_length_max,
        max_length_min=max_length_min,
        max_length_max=max_length_max,
        pause_on_max_length=pause_on_max_length,
        pause_on_min_length=pause_on_min_length,

        num_tubulus=num_tubulus,
        tubuli_min_dist=tubuli_min_dist,
        margin=margin,

        width_var_std=width_var_std,
        bend_amplitude=bend_amplitude,
        bend_prob=bend_prob,
        bend_straight_fraction=bend_straight_fraction,

        sigma_x=sigma_x,
        sigma_y=sigma_y,

        background_level=background_level,
        tubulus_contrast=tubulus_contrast,
        gaussian_noise=gaussian_noise,
        bleach_tau=bleach_tau,
        jitter_px=jitter_px,
        vignetting_strength=vignetting_strength,
        invert_contrast=invert_contrast,
        global_blur_sigma=global_blur_sigma,

        fixed_spot_count=fixed_spot_count,
        fixed_spot_intensity_min=fixed_spot_intensity_min,
        fixed_spot_intensity_max=fixed_spot_intensity_max,
        fixed_spot_radius_min=fixed_spot_radius_min,
        fixed_spot_radius_max=fixed_spot_radius_max,
        fixed_spot_kernel_size_min=fixed_spot_kernel_size_min,
        fixed_spot_kernel_size_max=fixed_spot_kernel_size_max,
        fixed_spot_sigma=fixed_spot_sigma,

        random_spot_count=random_spot_count,
        random_spot_intensity_min=random_spot_intensity_min,
        random_spot_intensity_max=random_spot_intensity_max,
        random_spot_radius_min=random_spot_radius_min,
        random_spot_radius_max=random_spot_radius_max,
        random_spot_kernel_size_min=random_spot_kernel_size_min,
        random_spot_kernel_size_max=random_spot_kernel_size_max,
        random_spot_sigma=random_spot_sigma,

        show_time=show_time,
        show_scale=show_scale,
        um_per_pixel=um_per_pixel,
        scale_bar_um=scale_bar_um,

    )

    synth_cfg.id = "current_trial"
    synth_cfg.validate()

    # Evaluate this new configuration against reference embeddings
    return run(synth_cfg, ref_embs, model, extractor)
