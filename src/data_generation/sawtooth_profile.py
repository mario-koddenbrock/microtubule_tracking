import matplotlib.pyplot as plt
import numpy as np


def create_sawtooth_profile(num_frames, max_length, min_length, noise_std, offset):
    profile = []
    grow_len = max_length
    shrink_len = max_length * 0.9
    grow_frames = int(num_frames / 6) + 1
    shrink_frames = int(grow_frames / 3) + 1
    t = 0
    while len(profile) < num_frames:

        # Grow phase (slow)
        for i in range(grow_frames):
            if len(profile) >= num_frames: break
            val = min_length + (i / grow_frames) * (grow_len - min_length) + np.random.normal(0, noise_std)
            profile.append(val)

        # Shrink phase (fast)
        for i in range(shrink_frames):
            if len(profile) >= num_frames: break
            val = min_length + (grow_len - min_length) * (1 - i / shrink_frames) + np.random.normal(0, noise_std)
            profile.append(val)

        profile = profile[:num_frames]

    profile = profile[offset:] + profile[:offset]
    # plt.plot(profile)
    # plt.title("Sawtooth Profile")
    # plt.xlabel("Frame Index")
    # plt.ylabel("Profile Value")
    # plt.grid()
    # plt.show()
    return profile