# Synthetic Data Generation

This document describes the usage of the `scripts/generate_synthetic_data.py` script for creating synthetic video datasets of fluorescent microtubules. The script is a flexible command-line interface (CLI) that generates videos, instance segmentation masks, and corresponding ground truth annotation files based on a JSON configuration.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Examples](#examples)
- [Configuration File](#configuration-file)
- [Output Files](#output-files)

## Prerequisites

Before running the script, ensure you have all the necessary Python packages installed. It is highly recommended to use a virtual environment.

1.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The script is run from the command line from the root directory of your project. It requires a base configuration file and an output directory, and you must specify which video IDs to generate.

### Command-Line Arguments

Here is a summary of the available arguments. You can always get this list by running `python scripts/generate_synthetic_data.py --help`.

| Argument                  | Shorthand | Required? | Description                                                                                             |
| ------------------------- | --------- | --------- | ------------------------------------------------------------------------------------------------------- |
| `--config <path>`         | `-c`      | **Yes**   | Path to the base `synthetic_config.json` file.                                                          |
| `--output-dir <path>`     | `-o`      | **Yes**   | Directory to save the generated videos, masks, and ground truth files. It will be created if it doesn't exist. |
| `--ids <id1> <id2> ...`   |           | **Yes**¹  | A space-separated list of specific video series IDs to generate.                                        |
| `--count <number>`        |           | **Yes**¹  | The number of videos to generate sequentially.                                                          |
| `--start-id <number>`     |           | No        | The starting ID for sequential generation. Used only with `--count`. (Default: `1`)                       |
| `--save-config`           |           | No        | Save a copy of the specific config file used for each video in the output directory for reproducibility.  |

¹ You must provide either `--ids` or `--count`, but not both.

### Examples

#### Generate a Sequence of Videos
To generate 10 videos sequentially (IDs 1 through 10) and save them to `data/generated/`:

```bash
python scripts/generate_synthetic_data.py \
    -c ./config/synthetic_config.json \
    -o ./data/generated \
    --count 10
```

## Installation

To get started, clone the repository and install the required dependencies. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/mario-koddenbrock/microtubule_tracking.git
cd microtubule_tracking

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```