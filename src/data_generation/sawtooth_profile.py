import time

import numpy as np
from scipy.ndimage import gaussian_filter1d


def create_sawtooth_profile(
    num_frames: int,
    max_length: int,
    min_length: int,
    grow_frames: int,
    shrink_frames: int,
    noise_std: float = 0.5,
    offset: int = 0,
    pause_on_min_length: int = 5,
    pause_on_max_length: int = 5,
):
    """
    Generate a single-cycle grow-⇢-shrink length profile with smooth transitions.
    """

    # number of frames per phase
    cycle_frames = grow_frames + shrink_frames

    # one cycle
    grow = np.linspace(min_length, max_length, grow_frames, endpoint=False)
    shrink = np.linspace(max_length, min_length, shrink_frames, endpoint=False)
    # add pauses at min and max lengths
    if pause_on_min_length > 0:
        shrink = np.concatenate([shrink, min_length * np.ones(pause_on_min_length, dtype=np.float32)])
    if pause_on_max_length > 0:
        grow = np.concatenate([grow, max_length * np.ones(pause_on_max_length, dtype=np.float32)])
    base_cycle = np.concatenate([grow, shrink])

    # tile cycles with twice the number of cycles to later change the starting point
    num_cycles = int(2 * np.ceil(num_frames / cycle_frames))
    profile = np.tile(base_cycle, num_cycles)

    # apply small **smoothed** noise using Gaussian filter
    noise = np.random.normal(0, noise_std, size=profile.shape)
    noise = gaussian_filter1d(noise, sigma=2)
    profile += noise

    # ensure the profile is long enough
    if len(profile) < (offset + num_frames):
        raise ValueError("Profile length is shorter than the number of frames requested.")

    # apply offset
    offset = offset % num_frames
    profile = profile[offset:offset + num_frames]

    # clamp to valid length range to prevent negative lengths
    profile = np.clip(profile, min_length, max_length)

    # plotting for debugging
    import matplotlib.pyplot as plt
    import os
    plt.plot(profile)
    plt.title("Sawtooth Profile")
    plt.xlabel("Frame Index")
    plt.ylabel("Length (px)")
    # save the plot to temporary folder with random name and timestamp
    plt.xlim(0, num_frames)
    plt.grid()
    plt.tight_layout()
    now = time.strftime('%Y%m%d-%H%M%S')
    os.makedirs("../.temp/sawtooth_profiles", exist_ok=True)
    plt.savefig(f"../.temp/sawtooth_profiles/{now}_{np.random.randint(10000)}.png")
    # plt.show()
    # close the plot
    plt.close()

    return profile.tolist()

