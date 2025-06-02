import numpy as np
from scipy.ndimage import gaussian_filter1d

def create_sawtooth_profile(
    num_frames: int,
    max_length: float,
    min_length: float,
    grow_freq: float,
    shrink_freq: float,
    noise_std: float = 0.5,
    offset: int = 0,
    fps: int = 25,
):
    """
    Generate a single-cycle grow-⇢-shrink length profile with smooth transitions.
    """
    # number of frames per phase
    grow_frames = max(1, int(round(fps / grow_freq)))
    shrink_frames = max(1, int(round(fps / shrink_freq)))
    cycle_frames = grow_frames + shrink_frames

    # one cycle
    grow = np.linspace(min_length, max_length, grow_frames, endpoint=False)
    shrink = np.linspace(max_length, min_length, shrink_frames, endpoint=False)
    base_cycle = np.concatenate([grow, shrink])

    # tile cycles to match num_frames
    num_cycles = int(np.ceil(num_frames / cycle_frames))
    profile = np.tile(base_cycle, num_cycles)[:num_frames]

    # apply small **smoothed** noise using Gaussian filter
    noise = np.random.normal(0, noise_std, size=profile.shape)

    noise = gaussian_filter1d(noise, sigma=2)
    profile += noise

    # apply circular offset
    offset = offset % num_frames
    profile = np.roll(profile, offset)

    # clamp to valid length range to prevent negative lengths
    profile = np.clip(profile, min_length, max_length)

    # # plotting for debugging
    # import matplotlib.pyplot as plt
    # plt.plot(profile)
    # plt.title("Sawtooth Profile")
    # plt.xlabel("Frame Index")
    # plt.ylabel("Length (px)")
    # plt.show()

    return profile.tolist()

