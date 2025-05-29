import numpy as np

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
    Generate a single-cycle grow-⇢-shrink length profile.

    Parameters
    ----------
    num_frames   : total frames you need
    max_length   : peak microtubule length  (pixels)
    min_length   : valley length            (pixels)
    grow_freq    : *Hz* — cycles/sec for growth       (lower = slower)
    shrink_freq  : *Hz* — cycles/sec for shrinkage
    noise_std    : Gaussian jitter (pixels)
    offset       : circular shift so every tubule starts at a different phase
    fps          : video frame-rate  (needed to convert Hz → frames)

    Notes
    -----
    • ``grow_freq`` and ``shrink_freq`` refer to *how fast a complete phase
      finishes*.  A smaller frequency → more frames spent in that phase.
    • If the requested profile is longer than *num_frames*, it repeats.
    """
    # number of frames per phase
    grow_frames   = max(1, int(round(fps / grow_freq)))
    shrink_frames = max(1, int(round(fps / shrink_freq)))

    profile = []

    while len(profile) < num_frames:
        # Grow (slow)
        for i in range(grow_frames):
            if len(profile) >= num_frames: break
            val = (min_length
                   + (i / grow_frames) * (max_length - min_length)
                   + np.random.normal(0, noise_std))
            profile.append(val)

        # Shrink (fast)
        for i in range(shrink_frames):
            if len(profile) >= num_frames: break
            val = (min_length
                   + (max_length - min_length) * (1 - i / shrink_frames)
                   + np.random.normal(0, noise_std))
            profile.append(val)

    # apply circular offset so every seed has a different starting phase
    offset = offset % num_frames
    profile = profile[offset:] + profile[:offset]
    return profile
