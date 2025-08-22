# Microtubule Tracking: Synthetic Data Generation

This document describes the usage of the `scripts/generate_synthetic_data.py` script for creating synthetic video datasets of fluorescent microtubules. The script is a flexible command-line interface (CLI) that generates videos, instance segmentation masks, and corresponding ground truth annotation files based on a JSON configuration.

## Table of Contents
- [Microtubule Tracking: Synthetic Data Generation](#microtubule-tracking-synthetic-data-generation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Command-Line Arguments](#command-line-arguments)
    - [Examples](#examples)
      - [Generate a single of Videos](#generate-a-single-of-videos)
      - [Generate a Sequence of Videos](#generate-a-sequence-of-videos)
    - [Configuration](#configuration)
  - [Contributing](#contributing)
  - [License](#license)

## Installation
To get started, clone the repository and install the required dependencies. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/mario-koddenbrock/microtubule_tracking.git
cd microtubule_tracking

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install package
pip install -e .
```

### Install micro_sam (required for some features)
`micro_sam` is only available via conda-forge. After installing the Python dependencies above, run:

```bash
conda install -c conda-forge micro_sam
```

Alternatively, you can use the provided `environment.yml` to set up everything with conda (see below).

## Usage

The script is run from the command line from the root directory of your project. It requires a base configuration file and an output directory, and you must specify which video IDs to generate.

### Command-Line Arguments

Here is a summary of the available arguments. You can always get this list by running `python scripts/generate_synthetic_data.py --help`.

| Argument                | Shorthand | Required? | Description                                                                                                    |
| ----------------------- | --------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| `--config <path>`       | `-c`      | **Yes**   | Path to the base `synthetic_config.json` file.                                                                 |
| `--output-dir <path>`   | `-o`      | **Yes**   | Directory to save the generated videos, masks, and ground truth files. It will be created if it doesn't exist. |
| `--ids <id1> <id2> ...` |           | No        | A space-separated list of specific video series IDs to generate.                                               |
| `--count <number>`      |           | No        | The number of videos to generate sequentially.                                                                 |
| `--start-id <number>`   |           | No        | The starting ID for sequential generation. Used only with `--count`. (Default: `1`)                            |
| `--save-config`         |           | No        | Save a copy of the specific config file used for each video in the output directory for reproducibility.       |



### Examples

#### Generate a single of Videos
To generate a video from `./config/synthetic_config.json` and save it to `data/generated/`:

```bash
python scripts/generate_synthetic_data.py \
    -c ./config/synthetic_config.json \
    -o ./data/generated \
    --count 1
```


#### Generate a Sequence of Videos
To generate 10 videos sequentially (IDs 1 through 10) and save them to `data/generated/`:

```bash
python scripts/generate_synthetic_data.py \
    -c ./config/synthetic_config.json \
    -o ./data/generated \
    --count 10
```


---

### Configuration

The synthetic video generation is controlled by a single JSON configuration file. The parameters are grouped by their effect on the final video, allowing for fine-grained control over the simulation.

| Effect / Feature            | Description                                                                                                                              | Relevant Parameters                                                                                                                      |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Core Video Properties**   | Basic settings for the output video dimensions, frame rate, and duration.                                                                | `id`, `img_size`, `fps`, `num_frames`, `color_mode`                                                                                      |
| **Microtubule Dynamics**    | Controls the stochastic growth and shrinkage behavior, mimicking dynamic instability. Sets the overall length constraints and pauses.    | `growth_speed`, `shrink_speed`, `catastrophe_prob`, `rescue_prob`, `min_length_min/max`, `microtubule_length_min/max`, `pause_on_min/max_length` |
| **Filament Structure**      | Defines the semi-flexible "train of wagons" model, controlling segment length and the potential for bending at junctions.                | `max_num_wagons`, `min/base_wagon_length_max`, `min/max_wagon_length_*`, `max_angle`, `max_angle_change_prob`                            |
| **Population & Seeding**    | Determines the number and initial placement of microtubules in the frame.                                                                | `num_microtubule`, `microtubule_min_dist`, `margin`                                                                                                |
| **Fluorescence & Staining** | Sets the colors and brightness of different microtubule parts, simulating fluorescent labels and tip-tracking proteins (+TIPs).          | `tubulus_contrast`, `seed_color_rgb`, `tubulus_color_rgb`, `tip_brightness_factor`                                                       |
| **Microscope Optics & PSF** | Simulates the Point Spread Function (PSF) and global blur, defining how sharp or blurry the microtubules appear.                         | `psf_sigma_h`, `psf_sigma_v`, `width_var_std`, `global_blur_sigma`                                                                               |
| **Camera & Illumination**   | Models the camera's background signal level, contrast mode, and common optical artifacts like uneven illumination (vignetting).          | `background_level`, `vignetting_strength`, `invert_contrast`                                                                             |
| **Camera Noise Model**      | Applies a realistic mixed noise model, combining signal-dependent photon shot noise (Poisson) and constant camera read noise (Gaussian). | `quantum_efficiency`, `gaussian_noise`                                                                                                   |
| **Other Realism Effects**   | Adds other common experimental artifacts like photobleaching (signal fading over time) and sample drift/jitter.                          | `bleach_tau`, `jitter_px`                                                                                                                |
| **Background Particles**    | Adds various types of spots to simulate cellular debris or other fluorescent particles. Each is an object with its own sub-parameters.   | `fixed_spots`, `moving_spots`, `random_spots`                                                                                            |
| **Frame Annotations**       | Overlays a timestamp and scale bar on the final video for easier analysis and presentation.                                              | `show_time`, `show_scale`, `um_per_pixel`, `scale_bar_um`                                                                                |
| **Output Settings**         | Controls whether to generate a pixel-perfect instance segmentation mask alongside the video, which is vital for training ML models.      | `generate_microtubule_mask`, `generate_seed_mask`                                                                                             |

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
