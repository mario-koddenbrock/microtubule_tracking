import numpy as np


def create_sawtooth_profile(num_frames, max_length, min_length, noise_std, offset):
    profile = []
    grow_len = max_length
    shrink_len = max_length * 0.9
    grow_frames = int(num_frames / 6)
    shrink_frames = int(grow_frames / 3)
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
    return profile


def add_gaussian(image, pos, sigma):
    x = np.arange(0, image.shape[1], 1)
    y = np.arange(0, image.shape[0], 1)
    x, y = np.meshgrid(x, y)
    gaussian = np.exp(-(((x - pos[0]) ** 2) / (2 * sigma[0] ** 2) +
                        ((y - pos[1]) ** 2) / (2 * sigma[1] ** 2)))
    image += gaussian
    return image


def normalize_image(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min + 1e-8)


def poisson_noise(image, snr):
    max_val = np.max(image)
    noisy = np.random.poisson(image * snr) / snr
    return np.clip(noisy / max_val if max_val > 0 else image, 0, 1)


def get_seed(img_size: tuple[int, int], margin: int):
    usable_width = img_size[1] - 2 * margin
    usable_height = img_size[0] - 2 * margin
    start_x = np.random.uniform(margin, margin + usable_width)
    start_y = np.random.uniform(margin, margin + usable_height)
    slope = np.random.uniform(-1.5, 1.5)
    intercept = start_y - slope * start_x

    return np.array([slope, intercept]), np.array([start_x, start_y])


def grow_shrink_seed(frame, original, slope, motion_profile, img_size: tuple[int, int], margin:int):
    net_motion = motion_profile[frame]

    dx = net_motion / np.sqrt(1 + slope ** 2)
    dy = slope * dx

    end_x = original[0] + dx
    end_y = original[1] + dy

    # Clip to safe margin
    end_x = np.clip(end_x, margin, img_size[1] - margin)
    end_y = np.clip(end_y, margin, img_size[0] - margin)

    return np.array([end_x, end_y])