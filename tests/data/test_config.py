import os
import tempfile

from data_generation.config import SyntheticDataConfig


def test_yaml_io():
    config = SyntheticDataConfig()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yml") as tmp:
        config.to_yml(tmp.name)
        tmp_path = tmp.name

    loaded_config = SyntheticDataConfig.from_yml(tmp_path)
    os.remove(tmp_path)

    assert config == loaded_config

def test_json_io():
    config = SyntheticDataConfig()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        config.to_json(tmp.name)
        tmp_path = tmp.name

    loaded_config = SyntheticDataConfig.from_json(tmp_path)
    os.remove(tmp_path)

    assert config == loaded_config

def test_update():
    config = SyntheticDataConfig()
    original_fps = config.fps
    config.update({"FPS": 30})
    assert config.fps == 30
    assert config.fps != original_fps
